import openai # OpenAIを使う場合
import google.generativeai as genai # Geminiを使う場合

import time
import datetime # Import the datetime module
import pytz    # Import the pytz module
import pandas as pd
from tqdm import tqdm
from google.colab import userdata
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

class llm_api:
  # 初期化
  # apikeyが無い場合は、評価なし
  def __init__(self, apikey, eval_model = "gpt", system_prompt = None):
    self.openai_api_key_ = apikey
    self.suffix_prompt_= ""
    self.input_prompt_hook_ = None
    self.system_prompt = system_prompt
    if apikey is None:
      print("not eval mode")
      self.eval_ai_model_ = "-"
      self.ai_client_ = None
      return

    if not eval_model.startswith("gemini"):
      # OpenAIのgpt系の場合
      self.use_gemini_ = False
      self.eval_ai_model_ = eval_model if eval_model != "gpt" else "gpt-4o-2024-11-20"
      self.ai_client_ = openai.OpenAI(api_key=self.openai_api_key_)
    else:
      # Google系の場合
      self.use_gemini_ = True
      self.eval_ai_model_ = eval_model if eval_model != "gemini" else "gemini-1.5-flash" # "gemini-1.5-pro"
      genai.configure(api_key=self.openai_api_key_)
      if self.system_prompt is not None:
        self.ai_client_ = genai.GenerativeModel(self.eval_ai_model_, system_instruction=self.system_prompt)
      else:
        self.ai_client_ = genai.GenerativeModel(self.eval_ai_model_)
    print(f"using {self.eval_ai_model_} ai model")
  pass

  # リトライ付きのOpenAIのAPIコール
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
  def completion_with_backoff(self, **kwargs):
    return self.ai_client_.chat.completions.create(**kwargs)

  # リトライ付きのGEMINIのAPIコール
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
  def gemni_completion_with_backoff(self, prompt):
    return self.ai_client_.generate_content(prompt)

  # 推論実行
  def inference(self, prompt, input_text = None):
    if input_text is None:
      text = prompt
    else:
      text = prompt.format(input_text=input_text)

    try:
      if self.use_gemini_:
        # GEMINI
        response = self.gemni_completion_with_backoff(text)
        # Check if the response contains a valid part before accessing response.text
        if response.candidates and response.candidates[0].content:
          output = response.text
        else:
          print("Warning: Gemini API did not return a valid response.")
          output = ""  # or some other default value        
      else:
        # OPEN_AI
        if self.system_prompt is not None:
          response = self.completion_with_backoff(
              model=self.eval_ai_model_,
              messages=[
                  {
                  "role": "developer",
                  "content": [ { "type": "text", "text": self.system_prompt }]
                  },
                  {
                  "role": "user", 
                  "content": [ {"type": "text", "text": text} ]
                  }],
              frequency_penalty=0,
              presence_penalty=0,
          )
        else:
          response = self.completion_with_backoff(
              model=self.eval_ai_model_,
              messages=[{"role": "user", "content": text}],
              frequency_penalty=0,
              presence_penalty=0,
          )
        output = response.choices[0].message.content

    except openai.RateLimitError as e:
      print(f"Rate limit exceeded. Retrying in a moment... Error: {e}")
    except e:
      print(f"Some what error: {e}")
      output = None
    return output
  pass
pass

