"""Module containing the names of the fields used in the table, to avoid typos."""

COUNT = 'count'
LOG_LIKELIHOOD = 'log_likelihood'
MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM = 'max_segmentation_log_likelihood_sum'
PMI_SCORE = 'pmi_score'


def get_token_field_name(token_i: int) -> str:
    return f'token_{token_i}'


def get_field_sql_type(field: str) -> str:
    """Returns the SQL type of the field."""
    if field.startswith('token'):
        return 'UINTEGER'
    if field == COUNT:
        return 'UINTEGER'
    if field in [LOG_LIKELIHOOD, MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM, PMI_SCORE]:
        return 'DOUBLE'
