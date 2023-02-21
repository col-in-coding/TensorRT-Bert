export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
python builder.py -x onnx/model.onnx -c bert-base-uncased/ -o engine/bert.plan -f | tee log.txt
