# test_my_unittest
from hydra import initialize, compose

# 1. initialize will add config_path the config search path within the context
# 2. The module with your configs should be importable.
#    it needs to have a __init__.py (can be empty).
# 3. THe config path is relative to the file calling initialize (this file)
from main import run_without_decorator


def test_with_initialize() -> None:
    common_command = ["hydra.runtime.output_dir='./'", "debug=1"]

    with initialize(version_base=None, config_path="config"):
        if False:  # unet
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 sg.params.condition_method=null dynamic=unetca_fast  sg.params.cond_drop_prob=1.0 sg.params.cond_dim=0 dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 sg.params.cond_scale=0  name=unet_c10_iter300k data.params.batch_size=128 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )
        elif False:  # unetc10, num0.1
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 data.num_samples=0.1 sg.params.condition_method=null dynamic=unetca_fast_c10  sg.params.cond_drop_prob=1.0 sg.params.cond_dim=0 dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=v1_unetc10_c10_iter300k_n0.1 data.params.batch_size=128 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # unet_c10
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 sg.params.condition_method=null dynamic=unetca_fast_c10  sg.params.cond_drop_prob=1.0 sg.params.cond_dim=0 dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=unetc10_c10_iter300k data.params.batch_size=128 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # unetc10, num0.1, translation,cutout
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 data.num_samples=0.1  sg.params.condition_method=auglevel sg.params.cond_dim=7 dynamic=unetca_fast_c10  sg.params.cond_drop_prob=1.0 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=v1_unetc10_c10_iter300k_n0.1_auglvl data.params.batch_size=128  debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )
        elif False:  # unetc10, num0.1,labelmix
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 data.num_samples=0.1  sg.params.condition_method=labelmix sg.params.cond_dim=10 dynamic=unetca_fast_222_32  sg.params.cond_drop_prob=1.0 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=unetc10_c10_iter100k_n0.1_labelmix data.params.batch_size=27  debug=0".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif True:  # unetc10, num0.1,label
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 data.num_samples=0.1  sg.params.condition_method=label sg.params.cond_dim=10 dynamic=unetca_fast_222_32  sg.params.cond_drop_prob=1.0 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=unetc10_c10_iter100k_n0.1_label data.params.batch_size=128  debug=0".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # unet, in32
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=in32_pickle sg.params.condition_method=null dynamic=unetca_fast  sg.params.cond_drop_prob=1.0 sg.params.cond_dim=0 dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 sg.params.cond_scale=0 data.trainer.max_epochs=20 name=in32_unet_ep20 data.params.batch_size=128 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # vis, afdog
            # +vis.random=1
            # +vis.chainvis=1
            # +vis.knn_vis=1
            # +vis.interp=1 +vis.interp_c.samples=5 +vis.interp_c.n=9
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=cifar10 data.num_samples=100 sg.params.condition_method=clipfeat   sg.params.cond_dim=512 dynamic=unetca_fast_222_32 dynamic.params.model_channels=64 sg.params.cond_drop_prob=1.0   model=ddpm_s32  dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=0 name=v1.4_unet222c64_32_c10_iter100k_n1000_clip data.params.batch_size=128 debug=0".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # unet, obama100, 0.01
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=fs_obama100  data.num_samples=1.0 sg.params.condition_method=auglevel  dynamic=unetca_fast_ffhq64 sg.params.cond_drop_prob=1.0 sg.params.cond_dim=7 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=unet_obama100_iter10k data.params.batch_size=48 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # unet_cfg, ffhq64, 0.01
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=ffhq64 data.num_samples=0.01 sg.params.condition_method=diffaug aug.aug_global_prob=0.5  dynamic=unetca_fast_cfg_ffhq64 sg.params.cond_drop_prob=0.1 sg.params.cond_dim=7 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=2   name=unet64_cfg_ffhq_iter150k_n0.01 data.params.batch_size=48 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # uvit, obama100, 0.01
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=fs_obama100  data.num_samples=1.0 sg.params.condition_method=auglevel  dynamic=uvit dynamic.params.patch_size=4 sg.params.cond_drop_prob=1.0 sg.params.cond_dim=7 dynamic.params.cond_token_num=1  sg.params.cond_scale=0   name=unet_obama100_iter10k data.params.batch_size=48 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # unet, FFHQ, 0.01
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=ffhq64  data.num_samples=0.01 sg.params.condition_method=null  model.params.sampling_val=ddim  model.params.num_timesteps_val=250 dynamic=unetca_fast_ffhq64 sg.params.cond_drop_prob=1.0 sg.params.cond_dim=0 dynamic.params.cond_token_num=0 dynamic.params.context_dim=32 sg.params.cond_scale=0   name=unet_ffhq_iter300k data.params.batch_size=48 debug=1".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        elif False:  # debug
            cfg = compose(
                config_name="config_base",
                return_hydra_config=True,
                overrides="data=ffhq64 data.num_samples=0.01 sg.params.condition_method=clipfeat feat.clipfeat.mix=mix  model.params.sampling_val=ddim model.params.num_timesteps_val=50 dynamic=unetca_fast_ffhq64 sg.params.cond_drop_prob=1.0 sg.params.cond_dim=513 dynamic.params.cond_token_num=1 dynamic.params.context_dim=32 sg.params.cond_scale=0 data.trainer.max_steps=100000 name=v1.3.3_unet64_obama100_iter100k_auglevel0.12_0.4_ddim50 data.params.batch_size=48 debug=0".replace(
                    "'", ""
                )
                .replace("~/", "")
                .split()
                + common_command,
            )

        print(cfg)
        run_without_decorator(cfg, run_unittest=True)


test_with_initialize()
