package io.github.flemmli97.rules;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RuleSet {

    public List<Rule<?>> rules;

    public Map<Attribute, Rule<?>> ruleMap = new HashMap<>();
}
