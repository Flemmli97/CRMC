package io.github.flemmli97.rules;

import io.github.flemmli97.dataset.Attribute;
import io.github.flemmli97.dataset.Data;

public class RuleImpl {

    int[] labels;

    Object[] predicates; //some function attribute -> boolean thing

    public boolean matches(Data data) {
        return false;
    }

    public record Rule(int x, boolean reverse) {

        public boolean match(Attribute att) {
            if (att.nominal) {
                if (this.reverse)
                    return att.val != this.x;
                else
                    return att.val == this.x;
            }
            if (this.reverse)
                return att.val > this.x;
            else
                return att.val <= this.x;
        }
    }
}
