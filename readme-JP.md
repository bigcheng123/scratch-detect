### Readme English [https://gitee.com/trgtokai/scratch-detect/blob/master/readme.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme.md)
### Readme 中国語 [https://gitee.com/trgtokai/scratch-detect/blob/master/readme-CN.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme-CN.md)
### Readme 日本語 [https://gitee.com/trgtokai/scratch-detect/blob/master/readme-JP.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme-JP.md)
…………………………………………………………………………………………………………………………………………………………………………
## プロジェクト名: [Scratch-detect] 傷跡検出

### 1. ヒント <br>

このプログラムは [YOLOv5 v6.1](https://github.com/ultralytics/yolov5/tree/v6.1) に基づいています。

**ハードウェア プラットフォーム:** <br>
カメラ: Hikvision MV-CU050-90UC<br>
PLC:三菱FX3S+modbusモジュール<br>
タッチスクリーン:威龍通MT8072IP<br>
光源: Hikvision リング光源 MV-LRSS-H-80-W<br>
光源コントローラー: [デジタル 8 チャンネル光源コントローラー](https://detail.tmall.com/item.htm?abbucket=1&id=656543446110&rn=21d65f2d271defe4d3b29e10ced9b2a5&spm=a1z10.5-b.w4011-23573612475.52.201646d6ZWIsQh&skuId=4738283905874)<br>

**PC推奨構成ハードウェア:**<br>
1. CPU: i7 13700k 以上<br>
2. グラフィックス カード: RTX3050 以降 (NVIDIA グラフィックス カードのみをサポート)<br>
3. メモリ: 16G 以上を推奨<br>
4. ハードドライブ: 1TB 以上を推奨<br>

**ソフトウェア プラットフォーム ソフトウェア：**<br>
1. システム: win10 x64 <br>
2. ドライバー: [hikvision MVS](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_4.3.2_240529.zip)、[WEINVIEW EBPRO](https://www.weinview.cn/Admin/Others/DownloadsPage.aspx?nid=3&id=10917&tag=0&ref=download&t=a4ff8b5703a191fe)、[NVIDIA グラフィックス ドライバー](https://cn.download.nvidia.com/Windows/555.99/555.99-desktop-win10-win11-64bit-international-nsd-dch-whql.exe)、[三菱 GXWORKS2](https://www.mitsubishielectric-fa.cn/site/file-software-detail?id=18) など<br>
3. Python環境：[anaconda](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Windows-x86_64.exe)<br>
4. PythonIDE: [PyCharm コミュニティ エディション](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC)<br>
5. Pythonバージョン：Python3.8<br>
6. バージョン管理: [Git](https://git-scm.com/download/win)

**AI 検出プログラム コード:**<br>
1. このプロジェクトのすべてのファイルをローカル コンピュータにクローン/ダウンロードします。
2. Pycharm または他の IDE ソフトウェアを使用して、このプロジェクトを開きます
3. 環境を構成し、すべての依存パッケージが正常にインストールされていることを確認します (manifest=requirement-trg.txt)。
4. メインプログラム main.py を実行します。
5. 独自の検出モデルを実行する場合は、トレーニング済みモデル ファイルを pt フォルダーに配置し、UI インターフェイスで対応する PT ファイルを選択してください。

### 2. デモビデオ デモ
中国ビリビリデモ ↓
[https://www.bilibili.com/video/BV1nz421S7KR](https://www.bilibili.com/video/BV1nz421S7KR)

中国国外での YouTube デモ ↓
[https://youtu.be/mEYHFr3ZQhM](https://youtu.be/mEYHFr3ZQhM)

### 3. インストール方法 インストール

1. ウェアハウス コードを複製する前に、anaconda、PyCharm、Git、その他のツールをインストールする必要があります<br>
2. 次のコードを使用してコードをローカルにクローンし、Python 環境を作成し、依存パッケージをインストールします。

```bash
git clone https://gitee.com/trgtokai/scratch-detect.git
cd scratch-detect
conda create -n yolov5_pyqt5 python=3.8
conda activate yolov5_pyqt5
pip install -r requirement-trg.txt
```

3. トーチおよびトーチビジョン ファイルはサイズが大きく、ダウンロードに時間がかかります。国内のソース (清華ソース/アリババ ソースなど) からダウンロードできます。
ダウンロードしたファイルを D:\code\scratch-detect\install_torch\ フォルダーに配置したら、次のコードを使用してインストールできます。

```bash
pip install -r requirement-torch Local Installation.txt
```

インストールが失敗した場合は、requirement-torch Local Installation.txt を開いて、ダウンロードしたファイルと一致するようにファイル名を変更します。<br>
4. インストール プロセス中に pycocotool コンパイル エラーが発生した場合は、Visual Studio C++ ビルド ツールをダウンロードしてインストールする必要があります<br>
5. 依存関係のインストールが完了したら、次のコードを使用してプログラムを実行します。

```bash
python main.py
```

### 4. 機能

1. 入力として画像、ビデオ、複数のカメラ、ネットワーク rtsp ストリーミングをサポート
2. ドロップダウン メニューでトレーニング モデルを変更します。
3. ボタンをドラッグして IoU を調整します
4. ボタンをドラッグして信頼度を調整します
5. 遅延を設定する
6. 開始、一時停止、および停止機能 (停止機能のバグを修正する必要があります)
7. 結果は Weintong タッチスクリーンにリアルタイムで統計的に表示されます。
8. 対象画像を認識後、自動保存
9. ModbusRTUプロトコルを使用してPLCと通信します
10.異常なターゲットが特定されたときに3色のライトとアラームをトリガーします
11. タッチスクリーンを使用して検出プログラムの起動を制御します
12. プログラム異常終了時の自動再起動チェック（PLCコイル信号が必要）

**実行インターフェイス:**
![画像の説明を入力](imgs/%E7%BA%BF%E4%B8%8A%E6%A3%80%E6%9F%A5%5B00_10_57%5D%5B20240605-174147%5D.png)

### 5. ファイル構成 ファイル

1. ./data/ - トレーニング スクリプト
2. ./pt/ - モデルファイルの保存場所
3. ./plc/ - PLC プロジェクト ファイルおよびタッチ スクリーン プロジェクト ファイル
4. ./runs/ - 実行結果の保存場所
5. ./ui_files/ - GUI ビジュアル インターフェイスのソース コードの保存場所
6. ./main.py - メインプログラム
7. ./train.py - モデルトレーニングプログラム
8. ./requirement-trg.txt - 依存関係パッケージのリスト
9. ./requirement-torch Local Installation.txt - ローカル依存パッケージのリスト

### 6. プログラム構造ネットワーク

 **プログラムの構造を次の図に示します。**
![プログラム構成図](imgs/%E7%A8%8B%E5%BA%8F%E7%BB%93%E6%9E%84%E5%9B%BE.png)

### 7. モデルトレーニング
1. [labelImg](https://blog.csdn.net/klaus_x/article/details/106854136) を使用して画像にラベルを付けます
2. データ内のトレーニング スクリプトのトレーニング セットの場所を変更します。詳細は [ここをクリック](https://blog.csdn.net/qq_45945548/article/details/121701492)
3. 独自の用途に適したモデル ファイルをトレーニングします。モデル ファイルを ./pt に配置して、メイン プログラム main.py の実行時に選択して使用します。

### 8. お問い合わせ
プログラムにバグやその他の提案がある場合は、このサイトを通じて @li-chey / @Alex_Kwan にプライベート メッセージを送信できます。