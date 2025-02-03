strings=("fidgety17" "fidgety29" "writhing17" "writhing29")
epoch=20
for str in "${strings[@]}"
do
    for cv in {6..10}
    do
    python tools/test.py configs/skeleton/gm/stgcn_2xb8-joint-${str}-keypoint-2d.py \
        work_dirs/${str}/${cv}/epoch_${epoch}.pth \
        --dump work_dirs/${str}/model_${cv}_e${epoch}.pkl
    done
done