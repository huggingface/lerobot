import ast
import grpc
import torch
import numpy as np
from transformers import DistilBertTokenizer
import io
import json

import lerobot.common.api.grpc.sac_pb2 as sac_pb2
import lerobot.common.api.grpc.sac_pb2_grpc as sac_pb2_grpc
from transformers import AutoModelForTokenClassification


class ModelConsumer:
    def __init__(self, model, tokenizer):
        # Initialize BERT model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup gRPC channel
        self.channel = grpc.insecure_channel("localhost:50051", options=[
                       ('grpc.max_receive_message_length', 4 * 1024 * 1024),  # 50 MB
                       ('grpc.max_send_message_length', 4 * 1024 * 1024),     # 50 MB
                       ('grpc.enable_retries',  1),
                       
                     ])
        self.stub = sac_pb2_grpc.TensorServiceStub(self.channel)

    def update_model_weights(self, bytes_buffer):
        """Update model weights from received state"""
        bytes_buffer.seek(0, io.SEEK_END)
        size_in_bytes = bytes_buffer.tell()

        # Reset buffer to the beginning
        bytes_buffer.seek(0)
        print(f"{size_in_bytes}")
        # Parse shapes dictionary from string
        try:
            state_dict = torch.load(bytes_buffer, weights_only=True)
            old_state_dict = self.model.state_dict()

            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
        
    def run_validation(self):
        """Run validation on current model state"""
        self.model.eval()
        
        # Example validation data
        text = "This is a great example"
        
        result = []
        with torch.no_grad():
            inputs = self.tokenizer(text,return_tensors="pt").to(self.model.device)
            
            outputs = self.model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            for p in predictions:
                result.append(self.tokenizer.decode(p))

        return result
            

    def start_consuming(self):
        print("Starting model consumer...")
        stream_request = sac_pb2.ModelRequest(model_name="bert_model")

        self.model.eval()

        epsilon = 0.01

        # Break the model
        for param in self.model.parameters():
            if param.requires_grad:  # ensure it's trainable
                # Example using uniform distribution near zero:
                param.data.uniform_(-epsilon, epsilon)

        print(f"Model broken")

        eval_info = self.run_validation()
        print(f"Evaluation results before receiving model update: {eval_info}")

        bytes_buffer = io.BytesIO()
        step = 0
        try:
            for model_update in self.stub.StreamModelUpdates(stream_request):
                if model_update.transfer_state == sac_pb2.TransferState.TRANSFER_BEGIN:
                    bytes_buffer.seek(0)
                    bytes_buffer.truncate(0)
                    bytes_buffer.write(model_update.data)
                    print("Received model update at step 0")
                    step = 0
                    continue
                elif model_update.transfer_state == sac_pb2.TransferState.TRANSFER_MIDDLE:
                    bytes_buffer.write(model_update.data)
                    step += 1
                    print(f"Received model update at step {step}")
                elif model_update.transfer_state == sac_pb2.TransferState.TRANSFER_END:
                    bytes_buffer.write(model_update.data)
                    print("Received model update at step end")
                    
                    # Update model weights
                    self.update_model_weights(bytes_buffer)

                    bytes_buffer.seek(0)
                    bytes_buffer.truncate(0)

                    print(f"Model updated")

                    # Run validation
                    eval_info = self.run_validation()
                    print(f"Evaluation results: {eval_info}")
                
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
        finally:
            self.channel.close()

id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

def init_model():
    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
    )
    return model

def init_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def main():
    model = init_model()
    tokenizer = init_tokenizer()
    model.to(torch.device("mps"))
    consumer = ModelConsumer(model, tokenizer)
    consumer.start_consuming()

if __name__ == "__main__":
    main()
