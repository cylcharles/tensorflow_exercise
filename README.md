# tensorflow_exercise
Using Tensorflow to do exercise.

## Linear regression
* 建立簡單Linear回歸模型

## Classfication
* 使用自訂義variables方法建立3層hidden layer
* 使用 Layer 方法建立3層hidden layer

## Simpson Classfication
* 分類辛普森家庭圖片
* 使用簡單5層Dense建立神經網路
* Note
  * 建立神經網路主要4步驟
    * Define placeholder
    * Create variables and operations (hidden layer)
    * Define loss function
    * Define Optimizer

## Simpson Classfication Cnn
* 分類辛普森家庭使用Convolutional Neural Networks(CNN)
* 讀取本地資料夾圖片
* 建立兩層卷積、池化 Layer
* 建立Fully-Connected Layer
* 建立輸出層 (6種類別)
* 優化器使用Adam進行優化
* import Keras ImageDataGenerator 來生成批量數據訓練模型
* Note
  * 最後Loss 及 Accuracy 結果
    * 觀察最後視覺化的結果，和只用5層Dense建立的神經網路比較，可看出Loss最終收斂接近於0，Accuracy最後也在接近1的地方，明顯比Dense的效果還要來得好
