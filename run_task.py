#%%
from litert_tools.pipeline import pipeline
from litert_tools.pipeline import task_file_processor as task_file_processor_lib
from litert_tools.pipeline import tokenizer as tokenizer_lib
from litert_tools.pipeline.pipeline import LiteRTLlmPipeline
from ai_edge_litert import interpreter as interpreter_lib
from typing import Optional
import sentencepiece as sp
#%%
def load(
      filename: str,
      tokenizer_location: Optional[str] = None,
  ) -> LiteRTLlmPipeline:

    try:
      if filename and filename.endswith(".task"):
        # Extract tflite, tokenizer and metadata from .task bundle
        file_processor = task_file_processor_lib.TaskFileProcessor(
            filename, cache_dir='cache'
        )
        model_path = file_processor.get_tflite_file_path()

        tokenizer_path = file_processor.get_tokenizer_file_path()
        raw_tokenizer = sp.SentencePieceProcessor()
        raw_tokenizer.Load(tokenizer_path)

        prompt_template = file_processor.get_prompt_template()
      else:
          raise ValueError(f"Unsupported file type: {filename}")
    except Exception as e:
      raise ValueError(f"Failed to load model from {filename}: {e}"
          "Failed to obtain tokenizer from %s: %s",
          tokenizer_location,
          e,
      )

    # Wrap the loaded tokenizer
    tokenizer = tokenizer_lib.Tokenizer(raw_tokenizer, prompt_template)

    # Load the interpreter
    print("Loading TFLite model from: %s", model_path)
    try:
      interpreter = interpreter_lib.InterpreterWithCustomOps(
          custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
          model_path=model_path,
          num_threads=2,  # Consider making num_threads configurable
          experimental_default_delegate_latest_features=True,
      )
    except Exception as e:
      raise ValueError(
          "Failed to load TFLite interpreter from %s: %s", model_path, e
      )
      raise

    # Create and return the pipeline with the wrapped tokenizer
    pipeline = LiteRTLlmPipeline(interpreter, tokenizer)
    print("LiteRTLlmPipeline loaded successfully.")
    return pipeline
#%%
runner = load('gemma3-1b-it-int4.task')

#%%
prompt = "สวัสดีครับ"
print(f'Prompt: {prompt} \n\n')
output = runner.generate(prompt, max_decode_steps=None)
print(f'Output: {output} \n\n')
#%%
