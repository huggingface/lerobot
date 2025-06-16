import { redirect } from "next/navigation";

export default function DatasetRootPage({
  params,
}: {
  params: { org: string; dataset: string };
}) {
  const episodeN = process.env.EPISODES
    ?.split(/\s+/)
    .map((x) => parseInt(x.trim(), 10))
    .filter((x) => !isNaN(x))[0] ?? 0;

  redirect(`/${params.org}/${params.dataset}/episode_${episodeN}`);
}
