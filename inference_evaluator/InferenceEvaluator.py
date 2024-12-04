
# InferenceEvaluatorのclass
# Elyza-task-100形式の評価データを使って自動評価するクラス
# Author: h.godai 2024/12/5
# 2024/12/9 prompt変更
# 2024/12/10 dpo eval用output記録
# 2024/12/10 promptを外部から与えられるように修正
# 2024/12/12 scoremapを追加
# 2024/12/12 suffixを追加
# 2024/12/13 command_prompt_を追加
# 2024/12/13 prefix/suffixをresultに書き出す
# 2024/12/15 Geminiに対応
# 2024/12/16 No evalに対応
# 参考資料
#
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
#
#

# Elaya-task-100の評価をする例
# ELYZA-tasks-100をダウンロードしてデータセットとして読み込み
#
# from datasets import load_dataset
#
# datasets = load_dataset("elyza/ELYZA-tasks-100")
#
# # 評価実行
# ie = InferenceEvaluator(model,            // モデル
#                         tokenizer,        // トークナイザ
#                         <OPENAI_API_KEY>  // OpenAIのAPI_KEY
#
# ie.run(dataset = datasets['test'],        // データセット (input ,output, eval_aspectが必須)
#        name = model_id,                   // 結果を保存する時に仕様するモデル名（テスト名）
#        max_tokens= 512,
# )
#
# ie.write_result()


import os
import time
import datetime # Import the datetime module
import pytz    # Import the pytz module
import time
import openai
import pandas as pd
from tqdm import tqdm
from google.colab import userdata
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import google.generativeai as genai

# 検証済みのモデル
# OpenAI
#  gpt-4o-2024-11-20        (default)
#  gpt-4o-mini-2024-07-18
# GEMINI
#  gemini-1.5-pro           (default)
#  gemini-1.5-flash



# 評価クラス定期
class InferenceEvaluator:
  model_ = None       # target LLM model
  model_name_ = ""
  tokenizer_ = None
  max_tokens_ = 512
  do_sample_ = False
  top_p_ = 0.9
  eval_count_ = 0
  temperature_ = 0
  repetition_penalty_ = 1.2
  eval_ai_model_ = "gpt-4o-2024-11-20"
  average_score_ = 0
  pasttime_ = 0
  ai_client = None
  output_result_ = []
  prefix_prompt_ = ""
  suffix_prompt_ = ""
  command_prompt_ = "### 指示\n"
  input_prompt_hook_ = None # few shot, RAG用の入力時のフック

  use_gemini_ = True
  openai_api_key_ = None

  # 初期化
  # apikeyが無い場合は、評価なし
  def __init__(self,  target_model, apikey, eval_model = "gpt"):
    self.model_ = target_model
    self.openai_api_key_ = apikey
    self.suffix_prompt_= ""
    self.input_prompt_hook_ = None
    if apikey is None:
      print("not eval mode")
      self.eval_ai_model_ = "-"
      self.ai_client_ = None
      return

    if eval_model.startswith("gpt"):
      # OpenAIのgpt系の場合
      self.use_gemini_ = False
      self.eval_ai_model_ = eval_model if eval_model != "gpt" else "gpt-4o-2024-11-20"
      self.ai_client_ = openai.OpenAI(api_key=self.openai_api_key_)
      print(self.ai_client_)
      self.eval_ai_model_ =  "gpt-4o-2024-11-20" # "gpt-4o-mini-2024-07-18" #
    else:
      # Google系の場合
      self.use_gemini_ = True
      self.eval_ai_model_ = eval_model if eval_model != "gemini" else "gemini-1.5-pro"
      genai.configure(api_key=self.openai_api_key_)
      #model = genai.GenerativeModel("gemini-1.5-flash")
      self.ai_client_ = genai.GenerativeModel("gemini-1.5-pro")
      self.eval_ai_model_ = "gemini-1.5-pro"
    print(f"using {self.eval_ai_model_} ai model")
  pass      


  # 推論の実行
  def inference(self, prompt):
    print(f"\n======= TASK-{self.eval_count_} :\n")
    inputtext = None
    if self.input_prompt_hook_:
      # プロンプティング用のフック
      inputtext = self.input_prompt_hook_(self, prompt)
    if inputtext is None:
      # デフォルトのプロンプト
      inputtext = f"{self.prefix_prompt_}{self.command_prompt_}{prompt}\n{self.suffix_prompt_}### 回答\n"
    print(inputtext)

    inputs = self.tokenizer_(inputtext, 
                             return_tensors="pt",
                             padding=True,
                             max_length=1024,        # 最大トークン長を設定（モデルの制限に合わせる）
                             truncation=True)

    if next(self.model_.parameters()).is_cuda:
        inputs = inputs.to("cuda")    # テキスト生成

    # 推論の開始
    output = self.model_.generate(
      inputs["input_ids"],
      max_new_tokens = self.max_tokens_,  # 最大生成トークン数
      do_sample=self.do_sample_,          # True, # False, #True,          # サンプリングを無効
      pad_token_id=tokenizer.pad_token_id,
      repetition_penalty=self.repetition_penalty_, # 繰り返しのペナルティ
      temperature = self.temperature_,    # 出力の多様性を制御
      top_p=self.top_p_                   # トップ確率のフィルタリング
    )

    # 結果をデコードして返す
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).split('\n### 回答')[-1]
    return generated_text.strip()

  # リトライ付きのOpenAIのAPIコール
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
  def completion_with_backoff(self, **kwargs):
    return self.ai_client_.chat.completions.create(**kwargs)

  # リトライ付きのGEMINIのAPIコール
  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
  def gemni_completion_with_backoff(self, prompt):
    return self.ai_client_.generate_content(prompt)

  # 評価実行
  def gpt4eval(self, pred, input_text, output_text, eval_aspect):
    prompt = f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{input_text}

# 正解例
{output_text}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{eval_aspect}

# 回答
{pred}
"""

    try:
      if self.use_gemini_:
        response = self.gemni_completion_with_backoff(prompt)
        gpt4score = response.text
      else:
        response = self.completion_with_backoff(
            model=self.eval_ai_model_,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
        )
        gpt4score = response.choices[0].message.content

    except openai.RateLimitError as e:
      # Handle rate limit errors gracefully
      print(f"Rate limit exceeded. Retrying in a moment... Error: {e}")
      # You might want to implement a more sophisticated retry mechanism here,
      # such as exponential backoff, to avoid hitting the rate limit repeatedly.
      time.sleep(60)  # Wait for 60 seconds before retrying
      # Retry the request
      if self.use_gemini_:
        response = self.gemni_completion_with_backoff(prompt)
        gpt4score = response.text
      else:
        response = self.completion_with_backoff(
            model=self.eval_ai_model_,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
        )
        gpt4score = response.choices[0].message.content
    try:
        gpt4score = int(gpt4score)
    except ValueError:
        print(f"Evaluation failed: {gpt4score}")
        gpt4score = None
    time.sleep(0.5) # APIコールに猶予を与えるため
    return gpt4score

  # 推論と評価
  def inference_and_eval(self, evaltext):
    if self.ai_client_ is None:
      return 0 # not eval mode
    start = time.time()  # 現在時刻（処理開始前）を取得
    generated_text = self.inference(evaltext['input']).strip()
    pasttime = (time.time()- start)
    self.pasttime_ += pasttime

    if len(generated_text) == 0:
      print(f"Inference empty result: {evaltext['input']}")
      score = 1
    score = self.gpt4eval(generated_text, evaltext["input"], evaltext["output"], evaltext["eval_aspect"])
    if score is None:
      score = 1
    print(f"\n回答\n{generated_text}\n=== SCORE: {score} ===\n")
    self.output_result_.append({"task_id": self.eval_count_,
                                "input": evaltext["input"],
                                "output": generated_text,
                                "score": score,
                                "example": evaltext["output"],
                                "inference time": f"{pasttime:.2f}"})
    return score

  # 推論と評価を開始する
  def run(self, dataset, name, tokenizer, max_tokens = 256,temperature = 0, repetition_penalty = 1.2, top_p = 0.9):
    self.model_name_ = name
    self.tokenizer_ = tokenizer
    self.max_tokens_ = max_tokens
    self.temperature_ = temperature
    self.repetition_penalty_ = repetition_penalty
    self.do_sample_ = temperature > 0
    self.top_p = top_p
    self.eval_count_ = 0
    self.output_result_ = []

    # ループ開始
    total_score = 0
    for dt in tqdm(dataset):
      score = self.inference_and_eval(dt) # lambda prompt : self.myllm(prompt), self.eval_llm_) # 実行
      total_score += score
      self.eval_count_ += 1
      self.average_score_ = total_score / self.eval_count_
      print(f"{self.eval_count_}/{len(dataset)} : score={score}, average={self.average_score_}")
    return self.average_score_
  pass

  # 結果を書き出す
  # pathnameで指定したファイルに結果を一行追加する
  # detail_pathで指定したフォルダに詳細な情報のファイルをexcelかcsv形式で格納する
  def write_result(self, pathname = "LLM2024_result.csv", detail_path = "./outputs", to_excel = True, to_csv = False):
    now = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime('%Y/%m/%d %H:%M:%S')
    scmap = [0,0,0,0,0]
    for row in self.output_result_:
      score = row['score']
      scmap[score-1] += 1
    print(scmap)
    resultstr=f"{self.model_name_},{self.temperature_},{self.max_tokens_},{self.average_score_},{self.eval_ai_model_},{now},{self.pasttime_:.1f},0,{scmap},{self.prefix_prompt_},{self.suffix_prompt_}\n"
    print(resultstr)
    if pathname is not None:
      #!touch $pathname
      with open(pathname, 'a', encoding='utf-8') as file:
        file.write(resultstr)
        file.close()

    if detail_path is not None:
      df = pd.DataFrame(self.output_result_)

      model_name = self.model_name_.replace("/", "_")
      now = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime('%Y%m%d_%H%M%S')

      if to_csv:
        # CSVファイルとして保存
        csv_file_path = f"{detail_path}/evalout_{now}_{model_name}.csv"
        df.to_csv(csv_file_path, index=False, encoding='utf-8')

      if to_excel:
        # エクセルファイルとして保存
        excel_file_path = f"{detail_path}/evalout_{now}_{model_name}.xlsx"
        df.to_excel(excel_file_path, index=False)
  pass

  # 単体試験用のテストコード
  # 使用例:
  #   InferenceEvaluator.test(model, tokenizer)
  def test(model, tokenizer):
    FastLanguageModel.for_inference(model)
    test = InferenceEvaluator(model, OPENAI_API_KEY) #userdata.get('OPENAI_API_KEY'))
    dataset = [{"input": "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。", "output":"努力、根性、気合、情熱、無心", "eval_aspect": "答えが5つあれば5点、それ以外は1点" },
               {"input": "日本で一番低い山は？", "output":"日和山", "eval_aspect": "日和山なら5点、さもなくば1点" }]
    test.run(dataset = dataset,
              name = "TEST_"+model.name_or_path,
              tokenizer = tokenizer,
              max_tokens= 256)
    test.write_result("InferenceEvaluatorTest", ".")

