import logging
import jinja2
from typing import Dict
from langchain.messages import HumanMessage, SystemMessage
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    Retrying
)

class StructuredLLMProcessor:
    """A generic class for processing LLM calls with structured inputs and outputs.
    It handles prompt templating, LLM invocation, and post-processing of results.
    """

    def __init__(self, name, llm, 
        instruction, description=None, prompt_defaults=None,
        post_processor=None, retry_configs={'max_attempts': 3, 'wait_base': 0.2, 'wait_min': 0, 'wait_max': 30}, 
        jinja2_env=None, logger=None
    ):
        """Create an instance of StructuredLLMProcessor.

        Args:
            name (str): The name of the processor.
            llm (BaseChatModel): The LLM to use for processing.
            instruction (str): The instruction template for the LLM.
            description (str, optional): The system prompt template for the LLM. Defaults to None.
            prompt_defaults (dict, optional): Default values for prompt variables. Defaults to None.
            post_processor (callable, optional): A function to process the LLM's output. The function 
                must accept a single argument (the LLM's output) and return a dictionary. Defaults to 
                a function that wraps the output in a dictionary with key "output".
            retry_configs (dict, optional): Configuration for tenacity retries. Allowed keys are 
                'max_attempts', 'wait_base', 'wait_min', 'wait_max'.
            jinja2_env (jinja2.Environment, optional): A Jinja2 environment to use for
                rendering prompts. If None, a default environment is created.
            logger (logging.Logger, optional): A logger instance. If None, a new logger is created.
        """

        if logger is None:
            logger = logging.getLogger(__name__)
        # end if
        self.logger = logger

        self.name = name
        self.llm = llm

        # set prompts
        self.instruction_prompt = instruction
        self.system_prompt = description

        if prompt_defaults is None:
            prompt_defaults = {}
        # end if
        self.prompt_defaults = prompt_defaults

        # create jinja2 environment
        if jinja2_env is None:
            jinja2_env = jinja2.Environment(
                loader     = jinja2.BaseLoader,
                undefined  = jinja2.StrictUndefined,
                autoescape = False
            )
        # end if
        self.jinja2_env = jinja2_env

        # set post processor
        if post_processor is None:
            post_processor = lambda x: {"output": x}
        # end if
        self.post_processor = post_processor

        # Retrying setting
        valid_retry_keys = {'max_attempts', 'wait_base', 'wait_min', 'wait_max'}
        retry_configs = {k: v for k, v in retry_configs.items() if k in valid_retry_keys}
        retry_settings = {
            'stop'  : stop_after_attempt(retry_configs.get('max_attempts', 1)),
            'wait'  : wait_exponential(
                exp_base = retry_configs.get('wait_base', 0.2), 
                min      = retry_configs.get('wait_min', 0), 
                max      = retry_configs.get('wait_max', 30)
            ),
            'retry' : retry_if_exception_type(Exception),
            'before': self._log_retry_attempt,
        }
        # - add retrying functionality to __call__
        self._retryer = Retrying(**retry_settings)
        self.__call__ = self._retryer.wraps(self.__call__)
    # end def

    def _log_retry_attempt(self, retry_state):
        """Helper function for logging before a retry sleep."""
        self.logger.warning(
            f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_wait_time:.2f}s "
            f"for {self.name} (Max attempts: {self.max_attempts})."
        )
    # end def


    def __call__(self, state, runtime=None, jinja2_env=None) -> Dict:
        """Processes the state, invokes the LLM, and post-processes the output.

        Args:
            state (dict): The current state of the graph.
            runtime (`Runtime`): The current runtime object
            jinja2_env (jinja2.Environment, optional): A Jinja2 environment to use for
                rendering prompts. If None, the processor's default environment is used.

        Returns:
            Dict: A dictionary containing the processed output.
        """

        if jinja2_env is None:
            jinja2_env = self.jinja2_env
        # end if

        # extra runtime variables
        # iterate through all attribute, choose only those name doesn't start with "_"
        runtime_var = {}
        if runtime is not None:
            for attr_name in dir(runtime.context):
                if attr_name.startswith("_"): continue
                attr_value = getattr(runtime.context, attr_name)
                if isinstance(attr_value, str):
                    runtime_var[attr_name] = attr_value
                # end if
            # end for
        # end if

        payloads = {**self.prompt_defaults, **state, **runtime_var}

        # make input messages to LLM
        input_messages = []
        if self.system_prompt:
            system_prompt = jinja2_env.from_string(self.system_prompt).render(**payloads)
            input_messages.append(SystemMessage(content=system_prompt))
        # end if
        instruction_prompt = jinja2_env.from_string(self.instruction_prompt).render(**payloads)
        input_messages.append(HumanMessage(content=instruction_prompt))

        # invoke LLM
        response = self.llm.invoke(input_messages).content
        # post process
        output = self.post_processor(response)
        return output
    # end def
# end class
