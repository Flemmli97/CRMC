package io.github.flemmli97.reflection;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

public class ReflectionUtil {

    private static final Map<String, Field> CACHE = new HashMap<>();

    @SuppressWarnings("unchecked")
    public static <T, I> T getField(I inst, String name) {
        String cacheName = inst.getClass().getPackageName() + "#" + name;
        Field cachedField = CACHE.get(cacheName);
        try {
            if (cachedField != null) {
                return (T) cachedField.get(inst);
            }
            Field field = inst.getClass().getDeclaredField(name);
            field.setAccessible(true);
            CACHE.put(cacheName, field);
            return (T) field.get(inst);
        } catch (IllegalAccessException | ClassCastException | NoSuchFieldException e) {
            System.out.println("Unable to get field " + name + " for " + inst);
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
