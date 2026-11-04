/** A cached response, if there is one. */
export declare function lookup(key: string): Promise<Response | undefined>;
/** Keeps `body` under `key` when the cache has room. */
export declare function store(key: string, body: BodyInit): Promise<void>;
/** A file from the cache, or downloaded (reporting progress in percent) and then kept. */
export declare function cached(url: string, progress?: (percent: number) => void): Promise<ArrayBuffer>;
