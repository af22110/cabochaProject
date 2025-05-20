from sudachipy import dictionary
from sudachipy import tokenizer

def check_word_in_sudachi_dictionary(word):
    """
    指定された単語がSudachiのシステム辞書に存在するかどうかを確認します。
    """
    # 辞書オブジェクトを直接作成し、lookupを呼び出す
    dict_obj = dictionary.Dictionary() # ここで辞書オブジェクトを作成
    
    # lookup() メソッドを使って、辞書に単語が存在するか検索
    # 戻り値は MorphemeList のリスト。単語が見つからなければ空のリスト。
    morphemes = dict_obj.lookup(word) # dictionaryオブジェクトからlookupを呼び出す
    
    return len(morphemes) > 0 # 見つかれば True, 見つからなければ False

def analyze_and_check_nouns_with_sudachi_dict(text):
    """
    SudachiPyで形態素解析を行い、名詞のみを抽出し、
    それらがSudachiのシステム辞書に正規語として存在するかを確認します。
    """
    # 形態素解析用のトークナイザーオブジェクト
    tokenizer_obj = dictionary.Dictionary(dict_type="core").create()
    mode = tokenizer.Tokenizer.SplitMode.B # 中単位分割

    results = {}
    morphemes = tokenizer_obj.tokenize(text, mode)

    for m in morphemes:
        pos = m.part_of_speech()
        if pos and pos[0] == '名詞':
            base_form = m.normalized_form()
            
            # ここでは、Sudachiのシステム辞書にその名詞が正規語として存在するかをチェックします
            is_in_dict = check_word_in_sudachi_dictionary(base_form)
            results[base_form] = is_in_dict
            
    return results

if __name__ == "__main__":
    input_text = "東京スカイツリーは日本の首都、東京にある展望台です。Pythonでプログラミング学習をしています。彼はミュージシンです。未知語テスト"

    print("--- Sudachiのシステム辞書による名詞の照合結果 ---")
    check_results = analyze_and_check_nouns_with_sudachi_dict(input_text)
    for noun, exists in check_results.items():
        if exists:
            print(f"'{noun}': Sudachi辞書に存在します")
        else:
            print(f"'{noun}': Sudachi辞書に存在しません（未知語または複合語の一部として扱われた可能性あり）")

    