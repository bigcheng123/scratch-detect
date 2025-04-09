
'''






'''



'''

GIT 放弃本地文件修改 ,将本地代码更新到与远程仓库一致
要放弃本地的文件修改，并将本地代码更新与远程仓库一致，可以使用以下步骤：

1.确保你已经添加了远程仓库（如果没有，请使用 git remote add origin <repository-url>）。

2.获取远程仓库的最新内容，但不自动合并到当前分支：
git fetch origin

3.重置当前分支到远程分支的状态（这将放弃所有本地未提交的更改）：
git reset --hard origin/masterils.general

注意：这里的 origin/master 应替换为你的远程分支名称，例如 origin/main 或者其他分支名。

4.如果你还需要丢弃所有未跟踪的文件和目录（例如，新添加的未添加到.git的文件），可以使用：
git clean -fd

请在执行这些操作之前确保你没有重要的本地更改，因为这些更改会被放弃。

# import apply_classifier
# from utils.datasets import LoadStreams, LoadImages
# 
# ################# 此文件为测试用草稿纸 ############################
# 
# function datasets():
#     class LoadStreams(source, img_size=imgsz, stride=stride)
#         def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
#     return self.sources, img, img0, None
# 
# dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     return self.sources, img, img0, None
# 
# function detect():
# for path, img, im0s, vid_cap in dataset:
# 
# def apply_classifier(x, model, img, im0):
#     return x
# pred = apply_classifier(pred, modelc, img, im0s)
# 
# # Process detections
# for i, det in enumerate(pred):  # detections per image
#     if webcam:  # batch_size >= 1
#         p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
#     else:
#         p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
# 
# 
# res = cv2.resize(im0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# cv2.imshow(str(p), res)  ##### show images
# 
# 
# pred = apply_classifier(pred, modelc, img, im0s)
# 
# for path, img, im0s, vid_cap in dataset:
'''