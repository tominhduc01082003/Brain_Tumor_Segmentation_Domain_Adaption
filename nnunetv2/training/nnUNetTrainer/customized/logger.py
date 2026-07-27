from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt

class My_nnUNetLogger(nnUNetLogger):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.my_fantastic_logging = {
            'mean_fg_dice_domain0': list(),
            'mean_fg_dice_domain1': list(),
            'ema_fg_dice_domain0': list(),
            'ema_fg_dice_domain1': list(),
            'dice_per_class_or_region_domain0': list(),
            'dice_per_class_or_region_domain1': list(),
            'train_losses': list(),
            'train_losses_s': list(),
            'train_losses_d': list(),
            'train_D_acc': list(),
            'val_D_acc': list(),
            'val_losses': list(),
            'lrs': list(),
            'grl_alphas': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.verbose = verbose
    def log(self, key, value, epoch: int):
        
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        if key == 'mean_fg_dice_domain0':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice_domain0'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice_domain0']) > 0 else value
            self.log('ema_fg_dice_domain0', new_ema_pseudo_dice, epoch)

        if key == 'mean_fg_dice_domain1':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice_domain1'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice_domain1']) > 0 else value
            self.log('ema_fg_dice_domain1', new_ema_pseudo_dice, epoch)
            
    def plot_progress_png(self, output_folder):
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1 
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 2, figsize=(60, 54))
        ax = ax_all[0, 0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr",
                linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val",
                linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice_domain0'][:epoch + 1], color='g', ls='dotted',
                 label="pseudo dice [Domain 0]",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice_domain1'][:epoch + 1], color='m', ls='dotted',
                 label="pseudo dice [Domain 1]",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice_domain0'][:epoch + 1], color='g', ls='-',
                 label="pseudo dice (mov. avg.) [Domain 0]",
                 linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice_domain1'][:epoch + 1], color='m', ls='-',
                 label="pseudo dice (mov. avg.) [Domain 1]",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        ax = ax_all[1, 0]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1],
                color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2, 0]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate",
                linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # S loss and D loss
        ax = ax_all[0, 1]
        ax.plot(x_values, self.my_fantastic_logging['train_losses_s'][:epoch + 1], color='b', ls='-', label="loss_s_tr",
                linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['train_losses_d'][:epoch + 1], color='r', ls='-', label="loss_d_tr",
                linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Train: loss_s and loss_d")
        ax.legend(loc=(0, 1))

        # grl_alpha
        ax = ax_all[1, 1]
        ax.plot(x_values, self.my_fantastic_logging['grl_alphas'][:epoch + 1], color='b', ls='-', label="grl_alpha",
                linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("grl_alphas")
        ax.legend(loc=(0, 1))

        # D accurate
        ax = ax_all[2, 1]
        ax.plot(x_values, self.my_fantastic_logging['train_D_acc'][:epoch + 1], color='b', ls='-', label="train_D_acc",
                linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_D_acc'][:epoch + 1], color='r', ls='-', label="val_D_acc",
                linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("D accuracy")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()