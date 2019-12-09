
## Supervision

add scope 'teacher/' to every variable in s0

merge s0 ckpt with pretrain weights

init from merged weights

combine three loss: kd_loss, mse_loss, ce_loss with alpha, beta, gamma 

