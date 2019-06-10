def discriminator_step(
    args, batch, generator_A2B, generator_B2A, discriminator_A2B, discriminator_B2A, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    #print(np.shape(obs_traj)) torch.Size([8, 737, 2])
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator_A2B(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator_A2B(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator_A2B(traj_real, traj_real_rel, seq_start_end)

    ########## added by cmb 还是要创建两个不同的generator
    reverse_seq_start_end = reverse_start_end_seq(seq_start_end)
    reverse_obs_traj = reverse(pred_traj_gt)
    reverse_obs_traj_rel = reverse(pred_traj_gt_rel)
    
    reverse_generator_out = generator_B2A(reverse_obs_traj, reverse_obs_traj_rel, reverse_seq_start_end)
    
    reverse_pred_traj_fake_rel = reverse_generator_out
    reverse_pred_traj_fake = relative_to_abs(reverse_pred_traj_fake_rel, reverse_obs_traj[-1])
    
    reverse_traj_real = reverse(traj_real)
    reverse_traj_real_rel = reverse(traj_real_rel)
    reverse_traj_fake = reverse(traj_fake)
    reverse_traj_fake_rel = reverse(traj_fake_rel)
    
    
    
    reverse_scores_fake = discriminator_B2A(reverse_traj_fake, reverse_traj_fake_rel, reverse_seq_start_end)
    reverse_scores_real = discriminator_B2A(reverse_traj_real, reverse_traj_real_rel, reverse_seq_start_end)
    
    # #print(reverse(traj_fake))
    # print(reverse_traj_fake.shape)
    

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    reverse_data_loss = d_loss_fn(reverse_scores_real, reverse_scores_fake)
    
    
    losses['D_data_loss'] = data_loss.item() + reverse_data_loss.item()
    loss += (data_loss + reverse_data_loss)
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator_A2B.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses