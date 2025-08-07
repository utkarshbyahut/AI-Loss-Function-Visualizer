import torch

def train_model(model, optimizer, loss_fn, X_train, y_train, epochs=100):
    loss_values = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    return model, loss_values
