"""Validate direct-vocabulary KD without fabricating a projection artifact."""


def validate_identity(student, teacher, policy_config, teacher_config):
    if student.get_vocab() != teacher.get_vocab() or len(student) != len(teacher):
        raise ValueError("Same-tokenizer mode requires identical token-to-ID mappings")
    if student.chat_template != teacher.chat_template:
        raise ValueError("Same-tokenizer mode requires identical local chat templates")
    if policy_config["tokenizer"]["chat_template_kwargs"] != teacher_config["tokenizer"]["chat_template_kwargs"]:
        raise ValueError("Same-tokenizer chat template kwargs differ")
    if teacher_config["max_total_sequence_length"] < policy_config["max_total_sequence_length"]:
        raise ValueError("Teacher context cannot accept the student sequence")
    if policy_config["make_sequence_length_divisible_by"] % teacher_config["make_sequence_length_divisible_by"]:
        raise ValueError("Teacher and student padding grids differ")


def projection_digest(path, digest):
    return None if path is None else digest(path)
