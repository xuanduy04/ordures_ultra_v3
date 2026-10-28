"""Dataset preprocessing selected only by the X-Token entrypoint."""

from nemo_rl.data.processors import filter_overlong_nemo_gym_dataset, nemo_gym_data_processor


def get_tokenizer(tokenizer_config, *, get_processor=None):
    """Apply X-Token loading kwargs without changing the company loader's defaults."""
    from nemo_rl.algorithms.utils import get_tokenizer as company_get_tokenizer
    from transformers import AutoProcessor, AutoTokenizer

    options = {} if get_processor is None else {"get_processor": get_processor}
    if "from_pretrained_kwargs" not in tokenizer_config or tokenizer_config["from_pretrained_kwargs"] is None:
        return company_get_tokenizer(tokenizer_config, **options)
    load_kwargs = dict(tokenizer_config["from_pretrained_kwargs"])
    if "trust_remote_code" in load_kwargs:
        raise ValueError("from_pretrained_kwargs must not duplicate trust_remote_code")
    loader = AutoProcessor if get_processor else AutoTokenizer

    def factory(name, **kwargs):
        return loader.from_pretrained(name, **kwargs, **load_kwargs)

    return company_get_tokenizer(tokenizer_config, pretrained_factory=factory, **options)


def dataset_transform(tokenizer, data_config):
    """Retain X-Token's reviewed length filter without changing GRPO callers."""
    maximum = data_config["max_input_seq_length"]

    def transform(data):
        if maximum is not None and data.processor is nemo_gym_data_processor:
            data.dataset = filter_overlong_nemo_gym_dataset(data.dataset, tokenizer, maximum)
        return data

    return transform
