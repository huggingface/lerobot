// Utility to post a message to the parent window with custom URLSearchParams
export function postParentMessageWithParams(
  setParams: (params: URLSearchParams) => void,
) {
  const parentOrigin = "https://huggingface.co";
  const searchParams = new URLSearchParams();
  setParams(searchParams);
  window.parent.postMessage(
    { queryString: searchParams.toString() },
    parentOrigin,
  );
}
