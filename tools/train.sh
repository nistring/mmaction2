for i in {1..10}
do
bash tools/dist_train.sh configs/skeleton/gm/stgcn_2xb8-joint-fidgety17-keypoint-2d.py 2 --work-dir work_dirs/fidgety17/${i}
bash tools/dist_train.sh configs/skeleton/gm/stgcn_2xb8-joint-fidgety29-keypoint-2d.py 2 --work-dir work_dirs/fidgety29/${i}
bash tools/dist_train.sh configs/skeleton/gm/stgcn_2xb8-joint-writhing17-keypoint-2d.py 2 --work-dir work_dirs/writhing17/${i}
bash tools/dist_train.sh configs/skeleton/gm/stgcn_2xb8-joint-writhing29-keypoint-2d.py 2 --work-dir work_dirs/writhing29/${i}
done