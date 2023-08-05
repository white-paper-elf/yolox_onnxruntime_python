## 本项目是用Python以onnxruntime的方式部署YOLOX的前向推理模型

### 关于如何利用onnx文件进行inference： 
在yolox_onnxruntime.py中的参数管理器（第317行的make_parser）中修改default=XXX的值,修改为自己的路径,修改好参数后直接点击yolox_onnxruntime_inference.py右键选择run即可运行。  
或者在终端运行运行

```python
python yolox_onnxruntime.py
```
