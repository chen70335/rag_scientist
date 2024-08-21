from typing import Optional

from ds_rag_framework.llms.prompts import PROMPTS_DICT


class PromptDesigner:
    """
    Class for implementing a system prompt to instruct LLMs in a custom way or with
    a predefined prompt that can be stored in the prompts.py file and called using this class.
    """
    @classmethod
    def get_prompt(
        cls,
        use_template: bool = True,
        template_name: Optional[str] = None,
        custom_full_prompt: Optional[str] = None,
    ) -> str:
        if use_template:
            if not template_name:
                raise ValueError(
                    "provide the name of the template in the prompts.py file"
                )
            return PROMPTS_DICT[template_name]
        elif not custom_full_prompt:
            raise ValueError(
                "full_prompt must be provided if use_template is False"
            )
        return custom_full_prompt
