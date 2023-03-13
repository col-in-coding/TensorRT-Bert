import onnx
import torch
import numpy as np
import onnxruntime as rt
import onnx_graphsurgeon as gs
from onnxsim import simplify
from onnx import shape_inference
from transformers import BertTokenizer
from torch.nn import functional as F


def test_onnx(onnx_path, BERT_PATH="bert-base-uncased/"):
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    input_ids = encoded_input['input_ids'].detach().numpy()
    token_type_ids = encoded_input['token_type_ids'].detach().numpy()
    attention_mask = encoded_input['attention_mask'].detach().numpy()
    position_ids = torch.arange(0, encoded_input['input_ids'].shape[1]).view(1, -1).numpy()

    onnx_model = rt.InferenceSession(onnx_path)
    onnxoutputs = onnx_model.run([], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    })
    print("output len: ", len(onnxoutputs))
    print(onnxoutputs[1].shape, f"sum: {onnxoutputs[1].sum()}")
    np.save("debug", onnxoutputs[1])

    logits = torch.from_numpy(onnxoutputs[0])
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)


if __name__ == "__main__":
    ONNX_PATH = "/workspace/Github/TensorRT-Bert/onnx/model.onnx"
    ONNX_SIM_PATH = "/workspace/Github/TensorRT-Bert/onnx/model-sim.onnx"
    ONNX_DEBUG_PATH = "/workspace/Github/TensorRT-Bert/onnx/model-sim-debug.onnx"

    # test_onnx(ONNX_DEBUG_PATH)
    # exit(0)

    onnx_model = onnx.load(ONNX_SIM_PATH)
    # model_simp, check = simplify(onnx_model)
    # onnx_model = shape_inference.infer_shapes(model_simp)
    # onnx.save(onnx_model, ONNX_SIM_PATH)

    # mark output for debug
    graph = gs.import_onnx(onnx_model)
    # print(graph.__dir__())
    # exit(0)
    for node in graph.nodes:
        # if node.name == "/bert/embeddings/LayerNorm/Add_1": # 验证Layernorm
        # if node.name == "/bert/encoder/layer.0/intermediate/dense/Add": # 验证AddLinear
        # if node.name == "/bert/encoder/layer.0/intermediate/intermediate_act_fn/Mul_1":  # 验证GELU
        # if node.name == "/bert/encoder/layer.0/output/LayerNorm/Add_1":  # 验证一层Transformer Layer
        if node.name == "/bert/encoder/layer.11/output/LayerNorm/Add_1":  # 最后一层Transformer Layer
            debug_tensor = node.outputs[0]
            # print(debug_tensor)
            # print(type(graph.outputs[0]))
            # print(type(graph.tensors()))
            graph.outputs.append(debug_tensor)
            # exit(0)
            break
    # print(graph.outputs)
    graph.cleanup().toposort()
    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, ONNX_DEBUG_PATH)

    test_onnx(ONNX_DEBUG_PATH)



