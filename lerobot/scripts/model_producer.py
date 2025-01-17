from concurrent import futures
import time
import grpc
import torch
import numpy as np
from transformers import BertTokenizer
from torch.optim import AdamW
import io

import lerobot.common.api.grpc.sac_pb2 as sac_pb2
import lerobot.common.api.grpc.sac_pb2_grpc as sac_pb2_grpc

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

MAX_MESSAGE_SIZE = 4 * 1024 * 1024 # 50 MB
CHUNK_SIZE = 2 * 1024 * 1024 # 10 MB

class TensorServiceServicer(sac_pb2_grpc.TensorServiceServicer):
    def __init__(self, model):
        # Initialize BERT model and tokenizer
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.training_step = 0
        
    def train_step(self):
        """Simulate a training step with dummy data"""
        self.model.train()
        
        # Dummy training data
        texts = [
            "This is a positive example",
            "This is a negative example"
        ]
        labels = torch.tensor([1, 0]).to(self.device)
        
        # Tokenize and prepare input
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def get_model_state(self):
        """Convert model state dict to flat array for transmission"""
        state_dict = self.model.state_dict()
        
        buffer = io.BytesIO()

        torch.save(state_dict, buffer)

        return buffer

    def StreamModelUpdates(self, request, context):
        print("Start model streaming")
        while True:
            # Train for some steps
            total_loss = 0
            # for _ in range(10):  # Update every 10 training steps
            #     loss = self.train_step()
            #     total_loss += loss
            #     self.training_step += 1
            
            avg_loss = total_loss / 10
            print(f"Step {self.training_step}, Average Loss: {avg_loss:.4f}")
            
            # Get current model state
            with self.get_model_state() as weights_io_buffer:
                # Calculate size of the buffer
                weights_io_buffer.seek(0, io.SEEK_END)
                size_in_bytes = weights_io_buffer.tell()

                # Reset buffer to the beginning
                weights_io_buffer.seek(0)

                sent_bytes = 0

                print(f"Model state size {size_in_bytes/1024/1024} MB with")

                while sent_bytes < size_in_bytes:
                    transfer_state = sac_pb2.TransferState.TRANSFER_MIDDLE

                    if sent_bytes == 0:
                        transfer_state = sac_pb2.TransferState.TRANSFER_BEGIN
                    elif sent_bytes + CHUNK_SIZE >= size_in_bytes:
                        transfer_state = sac_pb2.TransferState.TRANSFER_END

                    size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
                    chunk = weights_io_buffer.read(size_to_read)

                    yield sac_pb2.ModelState(
                        transfer_state=transfer_state,
                        data=chunk
                    )
                    sent_bytes += size_to_read
                    print(f"Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")
                
                print("Sent model update")
                time.sleep(10)  # Wait before next update

    def GetModel(self, request, context):
        state = self.get_model_state()
        return sac_pb2.ModelState(
            shapes=str(state["shapes"]),
            data=state["parameters"],
            step=state["step"]
        )

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
    
def serve():
    model = init_model()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                          options=[
                       ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),  # 50 MB
                       ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),     # 50 MB
                     ])
    sac_pb2_grpc.add_TensorServiceServicer_to_server(
        TensorServiceServicer(model), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Model producer server started...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
