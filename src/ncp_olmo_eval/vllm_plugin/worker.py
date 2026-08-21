"""vLLM worker hook that binds scheduler identity to ConceptLM state."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    def compile_or_warm_up_model(self) -> None:
        """Keep any eager kernel/model warmup outside request state."""

        model = self.model_runner.get_model()
        set_profile_mode = getattr(model, "set_profile_mode", None)
        if set_profile_mode is None:
            raise RuntimeError(
                "ConceptLMGPUWorker requires a model with set_profile_mode()"
            )
        set_profile_mode(True)
        try:
            super().compile_or_warm_up_model()
        finally:
            set_profile_mode(False)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "ModelRunnerOutput | None":
        model = self.model_runner.get_model()
        bind_scheduler_output = getattr(model, "bind_scheduler_output", None)
        if bind_scheduler_output is None:
            raise RuntimeError(
                "ConceptLMGPUWorker requires a model with bind_scheduler_output()"
            )
        bind_scheduler_output(scheduler_output, self.model_runner)
        return super().execute_model(scheduler_output)
