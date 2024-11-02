# ランダム打ち版
# コマンドライン版

import creversi
import random
import numpy as np
import time

import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

# ソフト名
print("id name 16-168reversi_mlp")
# 開発者名
print("id author R.Sueyoshi")

# ソースコードを書き換えれば後手番でも可能。
print("先手番専用AI")

# 学習モデル
nn_model_file = "nn_learnd_rev.pkl"


def com_mlp_think(board, clf):
  """
  MLPが思考する関数

  args:
      board: creversi用の局面データ
      clf: 機械学習した学習モデルのデータ

  return:
      best_move int: creversi用の打ち手の数値
      is_random bool: ランダム打ちしたらTrue
  """

  # 合法手を生成する。
  move_lst = list(board.legal_moves)  # 合法手をリスト化する

  # 学習データ
  # NumPy用配列の準備。2次元、8*8
  input_np = np.empty((1, 2, 8, 8), dtype=np.float32)
  # NumPy用配列input_npに現局面を入れる
  board.piece_planes(input_np)
  # 1次元化
  input_np = np.array(input_np.ravel())

  # 次の一手を予測する。
  best_index = np.argmax(clf.predict([input_np]))

  # 次の一手が合法手ではない場合はランダム打ち。
  if board.is_legal(best_index) == False:
    # 合法手をシャッフルする。
    is_random = True
    random.shuffle(move_lst)
    best_move = move_lst[0]
  else:
    is_random = False
    best_move = best_index

  # 次の一手を返す。
  return best_move, is_random


def display_waiting():
  """
  入力待ちを表示する関数

  args:
      なし

  return:
      なし
  """
  print("")
  print("あなた（x）の手番です。")
  print("次から選んで2文字（例 d3）で入力してください。")
  # 入力候補を表示する
  # 合法手を生成する。
  player_move_lst = list(board.legal_moves)  # 合法手をリスト化する
  print("（入力可能なもの）")
  for i in player_move_lst:
      print(creversi.move_to_str(i) + " ", end="")
  print("")
  print("「quit」を入力すれば終了します。")


# 局面にセットする。
board = creversi.Board()

# MLPの学習モデルを読み込む。
with open(nn_model_file, 'rb') as fp:
  clf = pickle.load(fp)

# 盤面を表示する。
print(board)

# 入力待ちを表示する
display_waiting()

illegal_count = 0  # 不正な手の数

# 入力待ち。
while True:
  input_cmd = input()

  # エンジン停止の合図
  if input_cmd == "quit":
    # コマンド受付を終了する。
    break

  # 入力されたコマンド（打ち手）
  try:
    move = creversi.move_from_str(input_cmd)
  # 想定外の入力値は、入力しなおしてもらう。
  except:
    print("無効な手です。もう一度入力してください。")
    continue

  # 合法手ではなかった場合は入力しなおしてもらう。
  if board.is_legal(move) == False:
    print("無効な手です。もう一度入力してください。")
    continue

  # 次の一手を表示する。
  #clear_output(True)
  print("あなた:", creversi.move_to_str(move))

  # 次の一手を打つ。
  board.move(move)

  # 盤面を表示する。
  print(board)
  print("")
  print("MLP-AIは、思考中です。")
  
  # 考えているふりをする。1～3秒だけ待つ。
  tink_time = random.randint(1, 3)
  time.sleep(tink_time)  
  print(f"MLP-AIは、{tink_time}秒考えました。")

  # 思考開始の合図
  # 自分の次の手を考えさせる。
  best_move, is_random = com_mlp_think(board, clf)

  # 次の一手を打つ。
  board.move(best_move)

  # 盤面を表示する。
  print("")
  print(board)

  # AIの次の一手を表示する。
  #clear_output(True)
  print("")
  if is_random == True:
    print("MLP-AIは、ランダム打ちしました。")
    illegal_count += 1
  print("MLP-AI:", creversi.move_to_str(best_move))

  # 終局判定
  if board.is_game_over() == True:
    print("")
    if board.diff_num() > 0:
      print(f"{board.piece_num()}対{board.opponent_piece_num()}で、あなたの勝ち!!")
      print(f"AIは{illegal_count}回ランダム打ちしました。")
    elif board.diff_num() == 0:
      print(f"{board.piece_num()}対{board.opponent_piece_num()}で、引き分け。")
      print(f"AIは{illegal_count}回ランダム打ちしました。")
    else:
      print(f"{board.piece_num()}対{board.opponent_piece_num()}で、あなたの負け。")
      print(f"AIは{illegal_count}回ランダム打ちしました。")
    break
  
  # 入力待ちを表示する
  display_waiting()
