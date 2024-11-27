from test_utils import *
from tqdm import tqdm
from model import AE
from mean_mpjpe import mpjpe_diffusion_all_min

human36_joints = {
    'pelvis': 0,
    'righthip': 1,
    'rightknee': 2,
    'rightfoot': 3,
    'lefthip': 4,
    'leftknee': 5,
    'leftfoot': 6,
    'spine1': 7,
    'neck': 8,
    'head': 9,
    'site': 10,
    'leftshoulder': 11,
    'leftelbow': 12,
    'leftwrist': 13,
    'rightshoulder': 14,
    'rightelbow': 15,
    'rightwrist': 16
}

human36_connections = {
    'pelvis': ['righthip', 'lefthip', 'spine1'],
    'righthip': ['rightknee'],
    'rightknee': ['rightfoot'],
    'lefthip': ['leftknee'],
    'leftknee': ['leftfoot'],
    'spine1': ['neck', 'leftshoulder', 'rightshoulder'],
    'neck': ['head', 'site'],
    'leftshoulder': ['leftelbow'],
    'leftelbow': ['leftwrist'],
    'rightshoulder': ['rightelbow'],
    'rightelbow': ['rightwrist']
}

human36_ord_list = [
    'righthip', 'lefthip', 'spine1', 'rightknee', 'rightfoot', 'leftknee', 'leftfoot',
    'neck', 'leftshoulder', 'rightshoulder', 'head', 'site', 'leftelbow', 'leftwrist',
    'rightelbow', 'rightwrist'
]

human36_bones = [
    ('pelvis', 'spine1'),
    ('spine1', 'neck'),
    ('neck', 'head'),
    ('neck', 'leftshoulder'),
    ('neck', 'rightshoulder'),
    ('leftshoulder', 'leftelbow'),
    ('leftelbow', 'leftwrist'),
    ('rightshoulder', 'rightelbow'),
    ('rightelbow', 'rightwrist'),
    ('pelvis', 'lefthip'),
    ('lefthip', 'leftknee'),
    ('leftknee', 'leftfoot'),
    ('pelvis', 'righthip'),
    ('righthip', 'rightknee'),
    ('rightknee', 'rightfoot')
]


if __name__ == '__main__':

    model_path = "../model/cartesian/"
    model_name = "AE_cartesian_hd128x64_ld32_test.pth"

    checkpoint = torch.load(model_path + model_name)
    coord_dim = checkpoint['coord_dim']
    angle_dim = checkpoint['angle_dim']
    hidden_dims = checkpoint['hidden_dim']
    latent_dim = checkpoint['latent_dim']
    attn_layers = checkpoint['attn_layers']

    model = AE(coord_dim, angle_dim, hidden_dims, latent_dim, attn_layers)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    ground_truth, predictions = load_diffusion_data()

    batch_size = 128
    total_batches = len(ground_truth) // batch_size + (1 if len(ground_truth) % batch_size != 0 else 0)

    mpjpe = []

    # Iterate over the array in chunks of [batch_size] elements
    for i in tqdm(range(0, len(ground_truth), batch_size), total=total_batches, desc="Processing", unit="batch"):

        # Select a batch of [batch_size] elements
        gt_batch = ground_truth[i:i + batch_size]
        pred_batch = predictions[i:i + batch_size]

        for batch_idx in range(len(gt_batch)):
            gt_sample = gt_batch[batch_idx]  # (1, 17, 3)
            hyp_for_curr_sample = pred_batch[batch_idx]  # (20, 17, 3)
            hyp_tensor = torch.from_numpy(hyp_for_curr_sample)
            pp_hyp = preprocess_batch(hyp_tensor)
            aggr_hyp = np.median(pp_hyp, axis=0, keepdims=True)  # (1, 17, 3)
            vae_input = torch.from_numpy(aggr_hyp[:, :-16])

            # * 1000 step is needed to amplify the small distances in the median hyp
            # / 1000 step is used to go back in the original scale
            _rec_pose = model(vae_input)
            _rec_pose_np = _rec_pose.detach().numpy()

            rec_input = np.concatenate((_rec_pose_np, aggr_hyp[:, -16:]), axis=1)
            pp_rec = postprocess_batch(torch.from_numpy(rec_input).float())

            # mm -> m
            predicted = pp_rec.unsqueeze(0).unsqueeze(0).unsqueeze(0) * 1000
            target = torch.from_numpy(gt_sample * 1000).unsqueeze(1)

            # predicted: ([1, 1, 1, 1, 12, 3])
            # target: ([1, 1, 12, 3])

            magic_error = mpjpe_diffusion_all_min(predicted, target, mean_pos=False).item()
            mpjpe.append(magic_error)

    mpjpe_sum = sum(mpjpe)
    mean_value = mpjpe_sum / len(mpjpe)
    print(f"Mean MPJPE: {round(mean_value, 4)}")
