class Policy:
    @classmethod
    def get_layer_rules(cls):
        try:
            return cls.layer_rules

        except AttributeError:
            return {}

    @classmethod
    def get_weight_rules(cls):
        try:
            return cls.weight_rules

        except AttributeError:
            return {}

    @classmethod
    def get_attr_rules(cls):
        try:
            return cls.attr_rules

        except AttributeError:
            return {}

    @classmethod
    def get_tie_rules(cls):
        try:
            return cls.tie_rules

        except AttributeError:
            return {}
