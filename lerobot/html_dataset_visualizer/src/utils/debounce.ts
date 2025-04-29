export function debounce<F extends (...args: any[]) => any>(
  func: F,
  waitFor: number,
): (...args: Parameters<F>) => void {
  let timeoutId: number;
  return (...args: Parameters<F>) => {
    clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => func(...args), waitFor);
  };
}
