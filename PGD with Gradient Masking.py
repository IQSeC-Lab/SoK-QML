# PGD with Gradient Masking

def pgd_attack(model, images, labels, eps=0.1, alpha=None, iters=40):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    ori_images = images.clone().detach()

    if alpha is None:
        alpha = eps / iters

    images.requires_grad = True
    model.eval()

    for _ in range(iters):
        outputs, _ = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images.requires_grad = True

    return images


# Gradient Masking Autograd Function
class GradientMaskingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        noise = torch.randn_like(grad_output) * 0.1  
        grad_input = grad_output + noise
        return grad_input

# MLP Model with Gradient Masking
class MLPWithGradientMasking(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h1 = F.relu(self.input_fc(x))
        h1 = GradientMaskingFunction.apply(h1) 
        h2 = F.relu(self.hidden_fc(h1))
        h2 = GradientMaskingFunction.apply(h2)  
        y_pred = self.output_fc(h2)
        return y_pred, h2



def generate_adversarial_data(model, data_loader, attack_func, epsilon, device, pgd_iters=40):
    model.eval()
    adv_images_list = []
    adv_labels_list = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_func(model, images, labels, eps=epsilon, iters=pgd_iters)
        adv_images_list.append(adv_images.detach().cpu())
        adv_labels_list.append(labels.detach().cpu())

    adv_images = torch.cat(adv_images_list, dim=0)
    adv_labels = torch.cat(adv_labels_list, dim=0)

    return adv_images, adv_labels



def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images.view(images.shape[0], -1))
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy



# Instantiate Masked Model
model_with_masking = MLPWithGradientMasking(INPUT_DIM, OUTPUT_DIM).to(device)

optimizer = optim.Adam(model_with_masking.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 5
train(model_with_masking, train_loader, test_loader, optimizer, criterion, device, epochs)



# Test adversarial robustness against PGD
epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies = []

for eps in epsilons:
    adv_test_images, adv_test_labels = generate_adversarial_data(
        model_with_masking, test_loader, pgd_attack, epsilon=eps, device=device, pgd_iters=40
    )

    adv_test_loader = DataLoader(
        torch.utils.data.TensorDataset(adv_test_images, adv_test_labels),
        batch_size=128,
        shuffle=False
    )

    adv_loss, adv_acc = evaluate(model_with_masking, adv_test_loader, criterion, device)
    accuracies.append(adv_acc) 

    print(f"[Masked Model] Epsilon: {eps:.1f} | Adversarial Accuracy: {adv_acc * 100:.2f}%")