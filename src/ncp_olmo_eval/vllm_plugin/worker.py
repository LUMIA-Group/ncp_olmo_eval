"""vLLM worker hook that binds scheduler identity to ConceptLM state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.v1.worker.gpu_worker import Worker

from .plugin import register as _register_conceptlm_backend

# vLLM imports the configured worker class inside each EngineCore subprocess.
# Register here as well as through the front-end plugin so those fresh child
# processes can resolve the custom architecture before loading model weights.
_register_conceptlm_backend()

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput


class ConceptLMGPUWorker(Worker):
    """GPU worker that exposes the current scheduler step to ConceptLM."""

    def init_device(self) -> None:
        """Install the pinned vLLM 0.13 DFlash runner before model loading."""

        super().init_device()
        from .model_runner import ConceptLMDFlashGPUModelRunner, dflash_vllm_013_enabled

        if not dflash_vllm_013_enabled():
            return
        if self.use_v2_model_runner:
            raise RuntimeError("NCP DFlash requires vLLM 0.13 model runner V1")
        # The upstream runner constructed by ``super`` has not loaded a model
        # or allocated KV cache yet. Replace it here so installed vLLM files do
        # not need to be patched in place.
        self.model_runner = ConceptLMDFlashGPUModelRunner(
            self.vllm_config,
            self.device,
        )

    def _bind_v2_input_batch(self, model: Any) -> None:
        """Install the vLLM >=0.25 V2 input-batch boundary exactly once."""

        if hasattr(self.model_runner, "input_batch"):
            return
        if getattr(self, "_conceptlm_input_batch_hook_installed", False):
            return
        model_state = getattr(self.model_runner, "model_state", None)
        prepare_inputs = getattr(model_state, "prepare_inputs", None)
        request_states = getattr(model, "request_states", None)
        if prepare_inputs is None or request_states is None:
            raise RuntimeError(
                "ConceptLMGPUWorker cannot bind the vLLM V2 input batch"
            )

        def prepare_inputs_with_conceptlm_state(
            input_batch: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            request_states.bind_input_batch(input_batch)
            return prepare_inputs(input_batch, *args, **kwargs)

        model_state.prepare_inputs = prepare_inputs_with_conceptlm_state
        self._conceptlm_input_batch_hook_installed = True

    def determine_available_memory(self) -> int:
        """Mark vLLM's allocation-only dummy forward as profile mode."""

        model = self.model_runner.get_model()
        set_profile_mode = getattr(model, "set_profile_mode", None)
        if set_profile_mode is None:
            raise RuntimeError(
                "ConceptLMGPUWorker requires a model with set_profile_mode()"
            )
        set_profile_mode(True)
        try:
            return super().determine_available_memory()
        finally:
            set_profile_mode(False)

    def compile_or_warm_up_model(self) -> Any:
        """Keep any eager kernel/model warmup outside request state."""

        model = self.model_runner.get_model()
        set_profile_mode = getattr(model, "set_profile_mode", None)
        if set_profile_mode is None:
            raise RuntimeError(
                "ConceptLMGPUWorker requires a model with set_profile_mode()"
            )
        set_profile_mode(True)
        try:
            # vLLM 0.13 returned None here, while vLLM >=0.25 returns a
            # CompilationTimes record consumed by the executor.  Preserve the
            # upstream result so the same worker hook remains compatible with
            # both versions.
            return super().compile_or_warm_up_model()
        finally:
            set_profile_mode(False)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "ModelRunnerOutput | None":
        model = self.model_runner.get_model()
        self._bind_v2_input_batch(model)
        bind_scheduler_output = getattr(model, "bind_scheduler_output", None)
        if bind_scheduler_output is None:
            raise RuntimeError(
                "ConceptLMGPUWorker requires a model with bind_scheduler_output()"
            )
        bind_scheduler_output(scheduler_output, self.model_runner)
        return super().execute_model(scheduler_output)
