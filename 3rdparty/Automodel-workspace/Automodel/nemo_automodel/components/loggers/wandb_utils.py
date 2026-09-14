
import logging


def suppress_wandb_log_messages():
    """
    Patches wandb logger to suppress upload messages.

    These occur usually on KeyboardInterrupt or program crash.

    To print the log url:
    run = wandb.init(...)
    print(run.url)
    """
    # (1) kill off all wandb logger output below CRITICAL
    logging.getLogger("wandb").setLevel(logging.CRITICAL)

    # (2) monkey‐patch any of the internal "_footer…" functions to no‐ops
    def _suppress_footer(*args, **kwargs):
        return None

    # Depending on your wandb version these lives under sdk.internal.file_pusher
    try:
        import wandb.sdk.internal.file_pusher as _fp

        for name in dir(_fp):
            if name.startswith("_footer"):
                setattr(_fp, name, _suppress_footer)
    except ImportError:
        pass

    # There is also a per‐run footer in
    # wandb.sdk.internal.run._footer_single_run_status_info
    try:
        import wandb.sdk.internal.run as _run

        if hasattr(_run, "_footer_single_run_status_info"):
            _run._footer_single_run_status_info = _suppress_footer
    except ImportError:
        pass
