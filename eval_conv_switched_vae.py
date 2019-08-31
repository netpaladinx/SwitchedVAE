def eval_print_code(eval_loader, model, device, path):
    checkpoint = torch.load(path)
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(path)
    with torch.no_grad():
        counts = defaultdict(lambda: defaultdict(int))
        for i, batch in enumerate(eval_loader):
            _, inputs = batch
            x = inputs.to(device).float()
            z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs = model.encoder(x)
            print('[Step %d - %d] %s, %s' % (step, i,
                                             ', '.join(['(%d, %.4f)' % (y_index.item(), z.item())
                                                    for y_index, z in zip(ys_index, zs)]),
                                             ', '.join([str(y_logits.cpu().numpy()) for y_logits in ys_logits])))
            for i1, y_index_1 in enumerate(ys_index):
                counts[i1][y_index_1.item()] += 1
                for i2, y_index_2 in enumerate(ys_index):
                    if i1 != i2:
                        counts[(i1, i2)][(y_index_1.item(), y_index_2.item())] += 1
        for k1 in counts:
            for k2 in counts[k1]:
                print(k1, k2, counts[k1][k2])


# def eval_visual(eval_loader, model, device, path, visual_dir):
#     checkpoint = torch.load(path)
#     step = checkpoint['step']
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     print(path)
#     with torch.no_grad():
#         for i, batch in enumerate(eval_loader):
#             _, inputs = batch
#             x = inputs.to(device).float()
#             recon_x, z2, z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs = model(x)
#             recon_x_from_mean = model.decoder(z2_mean, ys_index, ys_hard, zs)
#
#             # traversal over z2
#             traversal = z2_mean.new_tensor([-3, -2, -1, 0, 1, 2, 3])
#             n_latents = z2_mean.size(1)
#             n_traversal = traversal.size(0)
#             z2_mean_traversal = z2_mean.repeat(n_latents, n_traversal, 1)
#             for d in range(n_latents):
#                 z2_mean_traversal[d, :, d] = traversal
#             z2_mean_traversal = z2_mean_traversal.reshape(-1, n_latents)
#             recon_x_from_traversal = model.decoder(z2_mean_traversal,
#                                                    [y_index.repeat(z2_mean_traversal.size(0), 1) for y_index in ys_index])
#
#             images = torch.cat([recon_x_from_traversal, recon_x, recon_x_from_mean], 0).sigmoid()
#             images = torch.cat([images, x], 0)
#             visual_path = os.path.join(visual_dir, 'step-%d-image-%d-z.png' % (step, i))
#             plot_images(images, n_latents + 1, n_traversal, visual_path)
#
#             # traversal over y
#             yss_index = []
#             n_latents = len(ys_index)
#             n_branches = ys_hard[0].size(1)
#             for d in range(n_latents):
#                 for j in range(n_branches):
#                     ys_index_copy = [y_index.clone() for y_index in ys_index]
#                     ys_index_copy[d] = j
#                     yss_index.append(ys_index_copy)
#             ys_index_traversal = [torch.cat([ys_index[d] for ys_index in yss_index], 0)
#                                   for d in range(n_branches)]
#             recon_x_from_traversal = model.decoder(z2_mean.repeat(ys_index_traversal[0].size(0), 1),
#                                      ys_index_traversal)
#
#             images = torch.cat([recon_x_from_traversal, recon_x, recon_x_from_mean], 0).sigmoid()
#             images = torch.cat([images, x], 0)
#             visual_path = os.path.join(visual_dir, 'step-%d-image-%d-y.png' % (step, i))
#             plot_images(images, n_latents + 1, n_branches, visual_path)

def run_eval_print_code():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvSwitchedVAE(args.y_ce_beta, args.y_phsic_beta, args.y_mmd_beta, args.z_beta, args.z2_beta,
                            args.channels, args.n_branches).to(device)

    for path in os.listdir(args.save_dir):
        path = os.path.join(args.save_dir, path)
        if path[-5:] == '.ckpt':
            eval_loader, _ = get_data_loader(args.dataset, 1, 100)
            eval_print_code(eval_loader, model, device, path)


# def run_eval_visual(z_beta=10, y_beta=10, seed=1234):
#     save_dir = os.path.join(SAVE_DIR, DATASET_NAME)
#
#     torch.manual_seed(seed)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = FCSwitchedVAE(Y_CE_BETA, Y_PHSIC_BETA, Y_MMD_BETA, Z_BETA, Z2_BETA, IMG_CHANNELS).to(device)
#
#     visual_dir = os.path.join(OUTPUT_DIR, DATASET_NAME, 'visual')
#     if os.path.exists(visual_dir):
#         shutil.rmtree(visual_dir)
#     os.makedirs(visual_dir)
#
#     for path in os.listdir(save_dir):
#         path = os.path.join(save_dir, path)
#         if path[-5:] == '.ckpt':
#             eval_loader, _ = get_data_loader(DATASET_NAME, 1, 100)
#             eval_visual(eval_loader, model, device, path, visual_dir)