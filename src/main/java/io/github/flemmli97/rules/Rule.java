package io.github.flemmli97.rules;

import java.util.function.Predicate;

public interface Rule<T> extends Predicate<T> {

    boolean isNominal();

    default NominalRule asNominal() {
        if (this.isNominal())
            return (NominalRule) this;
        throw new IllegalStateException("Not a nominal rule");
    }

    default NumericRule asNumeric() {
        if (!this.isNominal())
            return (NumericRule) this;
        throw new IllegalStateException("Not a numeric rule");
    }

    record NominalRule(String attribute) implements Rule<String> {

        @Override
        public boolean test(String s) {
            return this.attribute.equals(s);
        }

        @Override
        public boolean isNominal() {
            return true;
        }
    }

    record NumericRule(int x) implements Rule<Double> {

        @Override
        public boolean test(Double v) {
            return this.x <= v;
        }

        @Override
        public boolean isNominal() {
            return false;
        }
    }
}
