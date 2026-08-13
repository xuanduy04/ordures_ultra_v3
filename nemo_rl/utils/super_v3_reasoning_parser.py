import logging

from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


logger = logging.getLogger(__name__)


# This will print when the module is imported/loaded
print("=" * 60)
print("SuperV3ReasoningParser plugin loaded and registered!")
print("=" * 60)


@ReasoningParserManager.register_module("super_v3")
class SuperV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )
        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and (
                request.chat_template_kwargs.get("enable_thinking") is False
                or request.chat_template_kwargs.get("force_nonempty_content") is True
            )
            and final_content is None
        ):
            """
            The original `deepseek_r1` reasoning parser this inherits from will automatically put everything in the reasoning content when it cannot parse out reasoning. This was fine for the DeepSeek R1 model that was not intended to be used without reasoning.
            1. Since the Nemotron 3 Nano and Super both have thinking off modes modulated by "enable_thinking=false" in the chat template kwargs, this change instead which will properly place the content in cases where there is no thinking enabled via config.
            2. There are rare cases where the model will output only reasoning without an end-think token `</think>` (e.g. reasoning exceeds max length), which results in empty content returned. End users may want to unilaterally avoid such cases and always have a content response even if the model does not finish its reasoning.
            """
            # Put all nonempty content into the content, rather than return content
            reasoning_content, final_content = None, reasoning_content

        elif final_content is not None:
            # Nemotron-3-Super's chat template renders assistant turns as
            #   "<think>\n" ~ reasoning ~ "\n</think>\n" ~ content
            # so the model emits "\n</think>\n" around the answer. The base
            # parser partitions on "</think>" without stripping, leaving the
            # template-owned trailing "\n" on reasoning and the leading "\n" on
            # content. Remove exactly one newline from each edge so that when
            # Gym re-renders multi-turn history, the template re-adds its own
            # newlines and the rendered string reproduces the generated token
            # sequence byte-for-byte. Without this, any agent making a 2nd model
            # call (e.g. structured_outputs_simple_agent, or any hallucinated
            # tool call) trips the token-monotonicity assert in
            # nemo_rl/environments/nemo_gym.py ("Non-contiguous messages found!").
            # removesuffix/removeprefix (single newline) instead of strip() so
            # genuine leading/trailing newlines inside reasoning/answer survive
            # the round-trip exactly.
            # NOTE: the swap branch above must not be stripped (thinking-off
            # generations are rendered without the think wrapper, so there is no
            # template newline to compensate) — hence elif, evaluated pre-swap.
            # Streaming extraction is not adjusted: NeMo Gym's policy model only
            # uses the non-streaming chat completions endpoint.
            if reasoning_content is not None:
                reasoning_content = reasoning_content.removesuffix("\n")
            final_content = final_content.removeprefix("\n")


        return reasoning_content, final_content