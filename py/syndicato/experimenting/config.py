"""
Configuration parameters
"""


__ARMS_DEFINITIONS__ = {
    "definitions": {
        "bernoulli": {
            "type": "object",
            "required": ["type", "name", "p", "reward"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["bernoulli"]
                },
                "name": {
                    "type": "string"
                },
                "p": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reward": {
                    "type": "number"
                }
            }
        }
    },
    "oneOf": [
        {"$ref": "#/definitions/arm-config/definitions/bernoulli"}
    ]
}

__ALGORITHMS_DEFINITIONS__ = {
    "definitions": {
        "epsilon-greedy": {
            "type": "object",
            "required": ["id", "configs"],
            "properties": {
                "id": {
                    "type": "string",
                    "enum": ["epsilon-greedy"]
                },
                "configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["epsilon"],
                        "properties": {
                            "epsilon": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        }
                    }
                }
            }
        },
        "softmax": {
            "type": "object",
            "required": ["id", "configs"],
            "properties": {
                "id": {
                    "type": "string",
                    "enum": ["softmax"]
                },
                "configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["temperature"],
                        "properties": {
                            "temperature": {
                                "type": "number",
                                "exclusiveMinimum": 0.0
                            }
                        }
                    }
                }
            }
        },
        "ucb1": {
            "type": "object",
            "required": ["id", "configs"],
            "properties": {
                "id": {
                    "type": "string",
                    "enum": ["ucb1"]
                },
                "configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["min-reward", "max-reward"],
                        "properties": {
                            "min-reward": {
                                "type": "number"
                            },
                            "max-reward": {
                                "type": "number"
                            }
                        }
                    }
                }
            }
        }
    },
    "oneOf": [
        {"$ref": "#/definitions/algorithm-config/definitions/epsilon-greedy"},
        {"$ref": "#/definitions/algorithm-config/definitions/softmax"},
        {"$ref": "#/definitions/algorithm-config/definitions/ucb1"}
    ]
}

__SCHEMA__ = {
    "$id": "https://thinkingthread.com/syndicato.schema.json",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Syndicato Experiments",
    "description": "Simulating context-free bandits",
    "definitions": {
        "algorithm-config": __ALGORITHMS_DEFINITIONS__,
        "arm-config": __ARMS_DEFINITIONS__,
        "report-config": {
            "type": "object",
            "properties": {
                "ci-scaling-factor": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            }
        },
        "simulation-config": {
            "type": "object",
            "required": ["runs", "trials"],
            "properties": {
                "runs": {
                    "type": "integer",
                    "minimum": 1
                },
                "trials": {
                    "type": "number",
                    "minimum": 1
                },
                "update-delay": {
                    "type": "integer",
                    "minimum": 0
                },
                "update-steps": {
                    "type": "integer",
                    "minimum": 1
                },
                "snapshot-steps": {
                    "type": "integer",
                    "minimum": 1
                }
            }
        }
    },
    "type": "object",
    "required": ["experiments"],
    "properties": {
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["algorithm", "config"],
                "properties": {
                    "algorithm": {
                        "$ref": "#/definitions/algorithm-config"
                    },
                    "arms": {
                        "type": "array",
                        "items": {
                            "$ref": "#/definitions/arm-config"
                        }
                    },
                    "config": {
                        "type": "object",
                        "required": ["report", "simulation"],
                        "properties": {
                            "report": {
                                "$ref": "#/definitions/report-config"
                            },
                            "simulation": {
                                "$ref": "#/definitions/simulation-config"
                            }
                        }
                    }
                }
            }
        }
    }
}


def validate_config(config):
    from jsonschema import validators
    _validator_class = validators.validator_for(__SCHEMA__)
    _validator_class.check_schema(__SCHEMA__)
    validator = _validator_class(__SCHEMA__)

    validator.validate(config)


def parse_config(path):
    import yaml

    with open(path, mode='r', encoding='utf-8') as fp:
        config = yaml.safe_load(fp)
    validate_config(config)
    return config
