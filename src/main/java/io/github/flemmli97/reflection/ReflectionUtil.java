package io.github.flemmli97.reflection;

import java.lang.reflect.Field;

public class ReflectionUtil {

    @SuppressWarnings("unchecked")
    public static <T, I> T getField(I inst, String name) {
        try {
            Field field = inst.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return (T) field.get(inst);
        } catch (IllegalAccessException | ClassCastException | NoSuchFieldException e) {
            System.out.println("Unable to get field " + name + " for " + inst);
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
