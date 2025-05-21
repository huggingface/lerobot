import { redirect } from "next/navigation";

export default function DatasetRootPage({
  params,
}: {
  params: { org: string; dataset: string };
}) {
  redirect(`/${params.org}/${params.dataset}/episode_1`);
  return null;
}
