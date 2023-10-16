model="clip_fusion"
dataset="MVSA_Single"
data_dir="./data/train_dataset/${dataset}"
nclasses=7
batch_size=32
max_epoch=40
save_path="./data/model/${model}/${dataset}"
lr=0.001
lr_decay=0.97
max_seq_length=64
image_encoder_pretrained_dir="./data/pretrain/ViT-B-16.pt"
output_dim=512
device_id=0

training_start_params="--model ${model} --dataset ${dataset} \
--data_dir ${data_dir} \
--nclasses ${nclasses} \
--batch_size ${batch_size} \
--max_epoch ${max_epoch} \
--save_path ${save_path} \
--lr ${lr} \
--lr_decay ${lr_decay} \
--max_seq_length ${max_seq_length} \
--image_encoder_pretrained_dir ${image_encoder_pretrained_dir} \
--output_dim ${output_dim} \
--bert_model_name ${bert_model_name} \
--bert_model_path ${bert_model_path} \
--device_id ${device_id}"

python train_classifier.py ${training_start_params}
