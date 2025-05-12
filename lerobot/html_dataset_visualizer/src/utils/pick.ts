/**
 * Return copy of object, only keeping whitelisted properties.
 *
 * This doesn't add {p: undefined} anymore, for props not in the o object.
 */
export function pick<T, K extends keyof T>(
  o: T,
  props: K[] | ReadonlyArray<K>,
): Pick<T, K> {
  // inspired by stackoverflow.com/questions/25553910/one-liner-to-take-some-properties-from-object-in-es-6
  return Object.assign(
    {},
    ...props.map((prop) => {
      if (o[prop] !== undefined) {
        return { [prop]: o[prop] };
      }
    }),
  );
}
