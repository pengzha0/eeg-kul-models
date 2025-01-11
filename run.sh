# "cacnn" "eegnet" "resnet18" "inception" "pyramidnet" "gcn" "cnn_lstm" "XGBoost" "Decision_Tree" "Random_Forest" "AdaBoost" "Gaussian_Naive_Bayes"

models=("cnn_lstm")

count=0
for model in "${models[@]}"; do
    for id in {0..15}; do  # 16个ID
        python main_left_one.py --subject_id $id --model $model
    done
done

wait  # 确保所有后台进程完成