# YOLO V2

1. 環境需求與建置
2. darkflow建置&測試
3. Use own dataset train a model 
4. 結論

## 1.環境需求與建置
#### 確認已具備以下環境：
* Python (I use Python3.7 with Anaconda)
* Tensorflow (I use Tensorflow-gpu 1.14)
* Opencv  
		有2種指令可以安裝，選哪一個都可以
   ##### cmd執行 
		pip install opencv-python		   (只安裝main model)
		pip install opencv-contrib-python  (main model and contrib model都安裝)

* Cython

  ##### cmd執行
		pip install Cython
####
* 若出現 "ImportError: No module named xxx"
  - 可能是你python環境中沒有此model
  - 到command line上執行: "pip install xxx"

## 2.darkflow建置&測試
* 我使用的Yolo V2是darkflow
  - use command line到你要放置的位置下執行，未來就在這個地方進行Yolo了
  ##### cmd執行
  		git clone https://github.com/thtrieu/darkflow
  - 建議挑選空間大的磁碟放置，未來訓練模型的weights檔會很多
  - 要在command line執行git指令要另外安裝Git並加入環境變數
  - 另外可以直接到網址內，以ZIP方式下載
* 安裝darkflow
  - command line進到darkflow資料夾內就可以進行安裝
  ##### cmd執行
  		pip install -e .
  - 以此指令安裝可以讓darkflow在globally dev mode. 讓我們可以在command line上直接運行
* darkflow測試
	* 先下載Pre-trained weights, 並放到darkflow資料夾下的bin資料夾內，若沒有bin自己創建
	#####  cmd下輸入:  
		python flow --model cfg/yolo.cfg --load bin/yolov2.weights --imgdir sample_img/
		說明：使用python執行, 執行flow接著輸入flow指令來運行你要做的事

		--model：取用你要使用的模型
		-- load：讀取對應模型的weights檔
		-- imgdir：要辨識圖片的位置
		
		更多指令可以輸入 python flow --h 來查找
	* 跑出來的結果會在sample_img資料夾下的out資料夾
	* 到這邊若沒有狀況,基本上就沒問題了,接著可以去訓練自己的要辨識的類別了

## 3.Use own dataset train a model 
### 前置作業：
1. Label image
	1. 使用軟體 **[labelImg](https://tzutalin.github.io/labelImg/)** (我是用Windows_v1.5.0)
	2. 安裝與操作可以參考這部影片[https://www.youtube.com/watch?time_continue=2&v=aiy9d9iS-3s](https://www.youtube.com/watch?time_continue=2&v=aiy9d9iS-3s)
	3. 要標記的圖片存放在"C:...\darkflow\test\training\images"
	4. 標記後的xml檔案設定存放在"C:...\darkflow\test\training\annotation" 
	5. 在label image時,路徑上建議不要有中文 


### Training on your own dataset:
#### 1. code參數修改
* 確定要分辨的類別數量,假設訓練類別(classes)只有1類, 這邊我要辨識的是柯南
	* 有幾個部分需要進行修改：
		##### 1. 你要使用的cfg模型檔, 假設是拿tiny-yolo-voc.cfg來進行訓練, 修改[region]中的classes(最後一層)
			...

			[region]
			anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
			bias_match=1
			classes=20    -->需要修改為1,分幾類就設定多少
			coords=4
			num=5
			softmax=1
			
			...
		##### 修改[convolutional]中的filter(倒數第2層),修改公式：5 * (classes number + 5), 假設classes=1, filter = 5 * (1 + 5) = 30
			...

			[convolutional]
			size=1
			stride=1
			pad=1
			filters=125   -->需要修改為30
			activation=linear

			[region]
			anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52

			...
		
		##### 2. 修改labels.txt內容 "C:...\darkflow\labels.txt" ,檔案是告知在訓練模型時的種類,要與label image時分類的種類數量與名稱相同, 記得每寫完一類別要進行換行
			Conan

#### 2. training conan model
1. 我在label image時只用了14張有柯南的照片, 因此只有14筆訓練資料, 下指令開始訓練
	* ##### cmd下指令
			python flow --model cfg/tiny-yolo-1c-train.cfg --load bin/yolov2-tiny-voc.weights --train --dataset test/training/images --annotation test/training/annotations --gpu 0.6
		##### 指令說明
			--model   	 :這邊我是選tiny-yolo-voc.cfg修改後儲存為tiny-yolo-1c-train.cfg進行訓練, 
			--load    	 :因此載入對應的weights是tiny-yolo-voc.weights
			注意：要載入正確的才不會報錯
			--dataset 	 :訓練集圖片資料夾位置
			--annotation :訓練集annotation(.xml)資料夾位置
			--gpu        :可設定0~1, 設定1就是全部記憶體資源都給他拿去跑, 我的顯卡普通因此設定1會出現CUDA Error: out of memory, 
						  若出現這種狀況就設定低一點的比例給training拿去跑,我這邊設定0.6(提供6成記憶體拿來訓練模型,我用的是GTX 960 4G)

2. 訓練時間約1小時train完1000epoch, avg loss降到1.xxx,再讓他繼續training,
	* 補充說明：
		* 每50個epoch就會產生4個檔案紀錄的是訓練時的weight和其他資訊,並存放在ckpt資料夾  
		* 但存放在ckpt的不是weights檔,而是其他4個類型檔案,需要指令轉換
	* ##### cmd下指令			
			python flow --model cfg/tiny-yolo-voc-1c.cfg --load -1 --train --dataset test/training/images/ --annotation test/training/annotations/ --gpu 0.6
		##### 指令說明
			--load -1 :會直接讀取ckpt中最新的weights和其他資訊進來繼續訓練
3. 一樣讓他跑完1000epoch，avg loss大約在800個epoch時就在0.3之間震盪，將最後的weights檔儲存為tensorflow格式 .pb & meta file 
	* ##### cmd下指令
			python flow --model cfg/tiny-yolo-voc-1c.cfg --load -1 --savepb
		##### 指令說明
			--savepb  :儲存為.pb & meta file

4. 最後進行辨識，將要辨識的圖片放入sample_img資料夾
	* ##### cmd下指令
			python flow --pbLoad built_graph/tiny-yolo-voc-1c.pb --metaLoad built_graph/tiny-yolo-voc-1c.meta --imgdir sample_img

## 4.結論
* 當初在label的資料使用的不多，導致在分辨上有的時候沒有辨識出是柯南
* 由於畫風緣故，會導致人物間有的臉部特徵很相似，導致誤認(像是把灰原辨識為柯南)
* 未來可嘗試加入更多筆資料來訓練，壓低avg loss
* 未來也可實驗特徵一致的事物來訓練，像電路板電容偵測好壞或是家庭、辦公室人員人臉辨識
* 此辨識方法，相較cosine similarity更精準，但相對的需要更多辨識者的圖像
