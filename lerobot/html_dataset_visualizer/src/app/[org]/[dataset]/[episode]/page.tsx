import EpisodeViewer from "./episode-viewer";
import { getEpisodeDataSafe } from "./fetch-data";

export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: {
  params: { org: string; dataset: string; episode: string };
}) {
  const { org, dataset, episode } = params;
  return {
    title: `${org}/${dataset} | episode ${episode}`,
  };
}

export default async function EpisodePage({
  params,
}: {
  params: { org: string; dataset: string; episode: string };
}) {
  // episode is like 'episode_1'
  const { org, dataset, episode } = params;
  // fetchData should be updated if needed to support this path pattern
  let episodeNumber = Number(episode.replace(/^episode_/, ""));
  if (episodeNumber) {
    episodeNumber--;
  }
  const { data, error } = await getEpisodeDataSafe(org, dataset, episodeNumber);
  return <EpisodeViewer data={data} error={error} />;
}
