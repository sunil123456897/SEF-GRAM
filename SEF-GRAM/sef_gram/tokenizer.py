class CharacterTokenizer:
    """
    Посимвольный токенизатор с фиксированным ASCII-260 словарем.
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.vocab_size = 260
        
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = []
        if add_bos:
            tokens.append(1)
        for char in text:
            code = ord(char)
            if code < 256:
                tokens.append(code + 4)
            else:
                tokens.append(3)
        if add_eos:
            tokens.append(2)
        return tokens
        
    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx < 4:
                if idx == 0:
                    continue  # Пропускаем паддинг
                elif idx == 1:
                    chars.append("[BOS]")
                elif idx == 2:
                    chars.append("[EOS]")
                elif idx == 3:
                    chars.append("[UNK]")
            else:
                chars.append(chr(idx - 4))
        return "".join(chars)
