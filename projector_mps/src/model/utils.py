import torch
import torch.nn.functional as F
import numpy as np


def pos_neg_infoNCE(image_embds, text_embds, labels, device, temperature=0.1):
    """
    Labels: Positive-Negative (0 or 1)
    Captions: 25 categories
    """
    batch_loss = []
    for i, label in enumerate(labels):
        pos_idxs = (labels == label).nonzero().squeeze()
        pos_idxs = pos_idxs[pos_idxs != i]
        neg_idxs = (labels != label).to(device)
        
        if pos_idxs.numel() == 0 or neg_idxs.sum() == 0:
            continue
        if pos_idxs.dim() == 0:
            pos_idxs = pos_idxs.unsqueeze(0)

        cos_sim_pos = F.cosine_similarity(image_embds[i].unsqueeze(0), text_embds[i].unsqueeze(0)) / temperature
        cos_sim_neg = F.cosine_similarity(image_embds[i].unsqueeze(0), text_embds[neg_idxs]) / temperature
        cos_sim_all = torch.cat([cos_sim_pos, cos_sim_neg], dim=0)
        
        exp_cos_sim = torch.exp(cos_sim_all)
        pos_exp_sum = exp_cos_sim[:pos_idxs.size(0)].sum()
        loss = -torch.log(pos_exp_sum / exp_cos_sim.sum())
        
        batch_loss.append(loss)
    
    if len(batch_loss) == 0:
        return torch.tensor(0.0, device=device)
    
    return torch.stack(batch_loss).mean()


def check_gradients(projector):
    gradients = []
    for name, parameter in projector.named_parameters():
        if parameter.grad is not None:
            grad_norm = parameter.grad.norm().item()
            gradients.append(grad_norm)
    return gradients


def get_combined_record(embeds):
    all_img_embeds = []
    all_labels = []
    all_sentiments = []
    for record in embeds: 
        all_img_embeds.append(record['img_embeds'])
        all_labels.append(record['labels'])
        all_sentiments += record['sentiments']
    all_img_embeds = torch.cat(all_img_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return {'img_embeds': all_img_embeds, 'labels': all_labels, 'sentiments': all_sentiments}


def compute_top_accuracy(model, test_img_embds):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    img_embeds = test_img_embds['img_embeds'].to(device)
    labels = test_img_embds['labels']
    sentiments = test_img_embds['sentiments']
    unique_sentiments = np.array(list(set(sentiments)))
    sentiments_embeds = model.open_elm.get_embds(unique_sentiments.tolist())
    sentiments_embeds = torch.nn.functional.normalize(sentiments_embeds, p=2, dim=1)

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    total_distances = []

    for i, q_emb in enumerate(img_embeds):
        gt_label = test_img_embds['sentiments'][i]
        distances = np.linalg.norm((sentiments_embeds - q_emb).cpu().detach().numpy(), axis=1)

        total_distances.append(np.min(distances))
        
        sorted_indices = np.argsort(distances).tolist()
        
        closest_labels_top1 = unique_sentiments[sorted_indices[:1]].tolist()[0]
        closest_labels_top3 = unique_sentiments[sorted_indices[:3]].tolist()[0]
        closest_labels_top5 = unique_sentiments[sorted_indices[:5]].tolist()[0]
        
        if gt_label in closest_labels_top1:
            correct_top1 += 1
        if gt_label in closest_labels_top3:
            correct_top3 += 1
        if gt_label in closest_labels_top5:
            correct_top5 += 1
        
        total += 1

    accuracy_top1 = correct_top1 / total
    accuracy_top3 = correct_top3 / total
    accuracy_top5 = correct_top5 / total

    mean_min_distance = np.mean(total_distances)

    print(f"Top-1 Accuracy: {accuracy_top1:.4f}")
    print(f"Top-3 Accuracy: {accuracy_top3:.4f}")
    print(f"Top-5 Accuracy: {accuracy_top5:.4f}")
    print(f"Mean Minimum Distance: {mean_min_distance:.4f}")

    return accuracy_top1

