from allennlp.data.fields import TextField


def text_field_to_text(text_field: TextField):
    return ' '.join(
        [item.text for item in text_field.tokens]
    )
