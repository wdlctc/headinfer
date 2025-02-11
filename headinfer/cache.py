from transformers.cache_utils import Cache, DynamicCache
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

class OffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()
        self.original_device = []
        self.prefetch_stream = torch.cuda.Stream()
        self.evit_stream = torch.cuda.Stream()
        self.beam_idx = None  # used to delay beam search operations

        self._seen_tokens = 0
        self._next_seen_tokens = 0

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            prev_layer_idx = (layer_idx - 1) % len(self)
            
            with torch.cuda.stream(self.evit_stream):
                self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
                self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)
                # self.key_cache[prev_layer_idx][:, :, self._next_seen_tokens:self._seen_tokens] = self.key_cache[prev_layer_idx][:, :, self._next_seen_tokens:self._seen_tokens].to("cpu", non_blocking=True)
                # self.value_cache[prev_layer_idx][:, :, self._next_seen_tokens:self._seen_tokens] = self.value_cache[prev_layer_idx][:, :, self._next_seen_tokens:self._seen_tokens].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._next_seen_tokens = self._seen_tokens
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None

